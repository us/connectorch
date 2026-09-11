"""Downloading and caching large dataset files.

The MaleCNS connectivity table is a gigabyte. Nothing here downloads a gigabyte
without the caller having said so, nothing re-downloads a file that is already
valid, and a download interrupted halfway leaves no half-file pretending to be
complete.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from ..exceptions import ConnectorchError

__all__ = ["default_cache_dir", "ensure_file"]

_CHUNK = 1 << 20


def default_cache_dir() -> Path:
    """Where datasets are cached: ``$CONNECTORCH_CACHE`` or ``~/.cache/connectorch``."""
    override = os.environ.get("CONNECTORCH_CACHE")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "connectorch"


def ensure_file(
    url: str,
    *,
    cache_dir: Path,
    filename: str | None = None,
    expected_size: int | None = None,
    download: bool | None = None,
    progress: bool = True,
) -> Path:
    """Return a local path to ``url``, downloading it only if allowed to.

    Parameters
    ----------
    url:
        Source to fetch.
    cache_dir:
        Directory the file is cached in. Created if missing.
    filename:
        Local name. Defaults to the last path segment of the URL.
    expected_size:
        Known size in bytes. A cached file of a different size is treated as
        incomplete and re-fetched; a partial download is resumed from its end.
    download:
        ``True`` downloads without asking. ``False`` never downloads and raises if
        the file is absent. ``None`` (the default) raises a message naming the
        exact size and URL, so nobody pulls a gigabyte by accident.

    Raises
    ------
    ConnectorchError
        If the file is missing and downloading was not authorised.
    """
    cache_dir = Path(cache_dir).expanduser()
    target = cache_dir / (filename or url.rsplit("/", 1)[-1])

    if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
        return target

    if download is False:
        raise ConnectorchError(
            f"{target} is not cached and download=False was passed.\n"
            f"Fetch it yourself from {url} and place it at {target}."
        )
    if download is None:
        size = f"{expected_size / 2**20:,.0f} MiB" if expected_size else "an unknown size"
        raise ConnectorchError(
            f"this needs {target.name} ({size}), which is not in {cache_dir}.\n"
            f"Source: {url}\n"
            "Pass download=True to fetch it, or download=False to fail fast when "
            "it is absent."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    _download(url, target, expected_size=expected_size, progress=progress)
    return target


def _download(
    url: str, target: Path, *, expected_size: int | None, progress: bool, resume: bool = True
) -> None:
    """Download to ``target.part``, resuming when it is safe to, then rename atomically.

    Resuming is only safe if the server confirms which bytes it is sending. A 206
    whose ``Content-Range`` does not start exactly where the local file ends means
    appending would interleave the wrong bytes, so the download restarts instead.
    Half a gigabyte of silently corrupted connectivity is worse than downloading
    it twice.
    """
    partial = target.with_suffix(target.suffix + ".part")
    already = partial.stat().st_size if partial.exists() else 0

    if expected_size is not None and already == expected_size:
        # A previous run finished the bytes but was killed before the rename.
        # Asking for bytes past the end would only earn a 416.
        shutil.move(str(partial), str(target))
        return

    if not resume:
        already = 0
    request = urllib.request.Request(url)
    if already:
        request.add_header("Range", f"bytes={already}-")

    try:
        response = urllib.request.urlopen(request)  # noqa: S310 (caller-supplied URL)
    except urllib.error.HTTPError as error:
        if error.code == 416 and already:
            # The server says our offset is past the end: the local file is stale
            # or already complete. Start over rather than guessing which.
            partial.unlink(missing_ok=True)
            return _download(
                url, target, expected_size=expected_size, progress=progress, resume=False
            )
        raise

    with response:
        if response.status == 206 and not _resume_confirmed(response, already):
            # A partial response we cannot place: its Content-Range is absent,
            # unparseable, or starts somewhere other than where our file ends. Its
            # body is a slice of unknown provenance, so it must neither be appended
            # nor mistaken for the whole file. This check runs whether or not we
            # asked for a range, because a 206 answering a rangeless request is
            # partial content all the same.
            partial.unlink(missing_ok=True)
            if not resume:
                raise ConnectorchError(
                    f"{url} answered a request for the whole file with partial "
                    f"content ({response.headers.get('Content-Range', 'no Content-Range')}). "
                    "Refusing to cache an incomplete file; retry later or fetch it "
                    "manually."
                )
            return _download(
                url, target, expected_size=expected_size, progress=progress, resume=False
            )
        if already and response.status != 206:
            already = 0
        declared = _declared_total(response, already, expected_size)
        with open(partial, "ab" if already else "wb") as handle:
            done = already
            while chunk := response.read(_CHUNK):
                handle.write(chunk)
                done += len(chunk)
                if progress:
                    _report(target.name, done, declared)
    if progress:
        _report(target.name, done, declared, final=True)

    # Measure before deleting: reporting the size of a file we just removed raises
    # FileNotFoundError and hides the real problem.
    actual = partial.stat().st_size
    wanted = expected_size if expected_size is not None else declared
    if wanted is not None and actual != wanted:
        partial.unlink(missing_ok=True)
        raise ConnectorchError(
            f"downloading {url} gave {actual:,} bytes but {wanted:,} were "
            "expected. The partial file was removed; retry."
        )
    shutil.move(str(partial), str(target))


def _parse_content_range(response) -> tuple[int, int] | None:
    """Return ``(first byte, total size)`` from a ``Content-Range`` header."""
    match = re.match(r"\s*bytes\s+(\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(3))


def _resume_confirmed(response, already: int) -> bool:
    """Whether the response really continues from byte ``already``.

    A plain 200 means the server ignored the Range header and is sending the
    whole file. A 206 whose ``Content-Range`` starts anywhere else means it is
    sending a different slice than the one asked for.
    """
    if response.status != 206 or already == 0:
        return False
    parsed = _parse_content_range(response)
    return parsed is not None and parsed[0] == already


def _declared_total(response, already: int, expected_size: int | None) -> int | None:
    """The file's full size, from the most authoritative source available.

    ``Content-Range`` carries the size of the *whole* file and is therefore
    trusted first. ``Content-Length`` only describes this response's body, so on a
    partial response it has to be added to what is already on disk; a server that
    sends a shorter range than asked for would otherwise look complete.
    """
    from_range = _parse_content_range(response)
    if from_range is not None:
        return from_range[1]
    header = response.headers.get("Content-Length")
    if header is None:
        return expected_size
    return already + int(header)


def _report(name: str, done: int, total: int | None, *, final: bool = False) -> None:
    if not sys.stderr.isatty() and not final:
        return
    if total:
        line = f"\r{name}: {done / 2**20:,.0f} / {total / 2**20:,.0f} MiB ({done / total:.0%})"
    else:
        line = f"\r{name}: {done / 2**20:,.0f} MiB"
    sys.stderr.write(line + ("\n" if final else ""))
    sys.stderr.flush()
