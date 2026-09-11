"""Download and resume behaviour, against a deliberately badly-behaved server.

The MaleCNS connectivity table is half a gigabyte. A resume that appends the wrong
bytes produces a file that is the right size, passes every later check, and is
quietly wrong. These tests exist because that failure is invisible otherwise.
"""

from __future__ import annotations

import http.server
import threading

import pytest

from connectorch import ConnectorchError
from connectorch.datasets._cache import default_cache_dir, ensure_file

BODY = b"abcde"


class AwkwardServer(http.server.BaseHTTPRequestHandler):
    """Serves ``BODY``, or misbehaves in a specific way, per :attr:`mode`."""

    mode = "normal"

    def log_message(self, *args: object) -> None:  # noqa: A003 - silence the test log
        pass

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
        if self.mode == "lying_range":
            self._send(206, b"ab", {"Content-Range": "bytes 0-1/5"})
        elif self.mode == "short_200":
            self.send_response(200)
            self.send_header("Content-Length", "5")
            self.end_headers()
            self.wfile.write(b"abc")
        elif self.mode == "short_range":
            # Answers the resume, but sends fewer bytes than the file has.
            self._send(206, b"d", {"Content-Range": f"bytes 3-3/{len(BODY)}"})
        elif self.mode == "unplaceable_range":
            # A partial response whose Content-Range has no total: it cannot be
            # placed, and its body is not the whole file either.
            if self.headers.get("Range"):
                self._send(206, b"de", {"Content-Range": "bytes 3-4/*"})
            else:
                self._send(200, BODY)
        elif self.mode == "always_unplaceable":
            # Never answers with the whole file, however it is asked.
            self._send(206, b"de", {"Content-Range": "bytes 3-4/*"})
        elif self.mode == "always_416":
            self._send(416, b"", {"Content-Range": f"bytes */{len(BODY)}"})
        else:
            self._serve_range()

    def _serve_range(self) -> None:
        header = self.headers.get("Range")
        if not header:
            self._send(200, BODY)
            return
        start = int(header.split("=")[1].split("-")[0])
        if start >= len(BODY):
            self._send(416, b"", {"Content-Range": f"bytes */{len(BODY)}"})
            return
        self._send(
            206,
            BODY[start:],
            {"Content-Range": f"bytes {start}-{len(BODY) - 1}/{len(BODY)}"},
        )

    def _send(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)


@pytest.fixture
def server():
    httpd = http.server.HTTPServer(("127.0.0.1", 0), AwkwardServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    AwkwardServer.mode = "normal"
    yield f"http://127.0.0.1:{httpd.server_port}/f.bin"
    httpd.shutdown()
    httpd.server_close()


def fetch(url: str, cache_dir, **kwargs):
    return ensure_file(url, cache_dir=cache_dir, download=True, progress=False, **kwargs)


def test_plain_download(server, tmp_path) -> None:
    assert fetch(server, tmp_path, expected_size=5).read_bytes() == BODY


def test_a_cached_file_is_not_fetched_again(server, tmp_path) -> None:
    fetch(server, tmp_path, expected_size=5)
    AwkwardServer.mode = "short_200"  # would corrupt the file if it were re-fetched
    assert fetch(server, tmp_path, expected_size=5).read_bytes() == BODY


def test_a_genuine_resume_completes_the_file(server, tmp_path) -> None:
    (tmp_path / "f.bin.part").write_bytes(b"abc")
    assert fetch(server, tmp_path, expected_size=5).read_bytes() == BODY


def test_a_server_resuming_from_the_wrong_offset_never_publishes_a_file(server, tmp_path) -> None:
    """A 206 whose Content-Range disagrees must not be appended to the local file."""
    (tmp_path / "f.bin.part").write_bytes(b"abc")
    AwkwardServer.mode = "lying_range"
    with pytest.raises(ConnectorchError):
        fetch(server, tmp_path, expected_size=5)
    assert not (tmp_path / "f.bin").exists()


def test_a_truncated_response_is_caught_without_an_expected_size(server, tmp_path) -> None:
    """Content-Length is the only size the caller gave us; check against it."""
    AwkwardServer.mode = "short_200"
    with pytest.raises(ConnectorchError, match="gave 3 bytes but 5"):
        fetch(server, tmp_path)
    assert not (tmp_path / "f.bin").exists()


def test_a_complete_partial_file_is_recovered_not_wedged(server, tmp_path) -> None:
    """Interrupted after the last byte but before the rename: the retry must finish it."""
    (tmp_path / "f.bin.part").write_bytes(BODY)
    AwkwardServer.mode = "always_416"
    assert fetch(server, tmp_path, expected_size=5).read_bytes() == BODY


def test_a_stale_oversized_partial_file_restarts(server, tmp_path) -> None:
    """A .part longer than the real file earns a 416; start over rather than wedge."""
    (tmp_path / "f.bin.part").write_bytes(b"abcdefghij")
    assert fetch(server, tmp_path, expected_size=5).read_bytes() == BODY


def test_a_short_range_response_is_not_mistaken_for_a_complete_file(server, tmp_path) -> None:
    """Content-Range carries the whole file's size; Content-Length only this slice."""
    (tmp_path / "f.bin.part").write_bytes(b"abc")
    AwkwardServer.mode = "short_range"
    with pytest.raises(ConnectorchError, match="gave 4 bytes but 5"):
        fetch(server, tmp_path)
    assert not (tmp_path / "f.bin").exists()


def test_an_unplaceable_partial_response_is_never_published(server, tmp_path) -> None:
    """A 206 we cannot place must not be appended, nor mistaken for the whole file."""
    (tmp_path / "f.bin.part").write_bytes(b"abc")
    AwkwardServer.mode = "unplaceable_range"
    assert fetch(server, tmp_path).read_bytes() == BODY


def test_a_server_that_only_ever_sends_partial_content_is_refused(server, tmp_path) -> None:
    """The retry must be validated too; the first fix only checked the resume path."""
    (tmp_path / "f.bin.part").write_bytes(b"abc")
    AwkwardServer.mode = "always_unplaceable"
    with pytest.raises(ConnectorchError, match="partial content"):
        fetch(server, tmp_path)
    assert not (tmp_path / "f.bin").exists()


def test_nothing_is_downloaded_without_permission(server, tmp_path) -> None:
    with pytest.raises(ConnectorchError, match="Pass download=True"):
        ensure_file(server, cache_dir=tmp_path, expected_size=5, progress=False)
    with pytest.raises(ConnectorchError, match="download=False"):
        ensure_file(server, cache_dir=tmp_path, expected_size=5, download=False, progress=False)


def test_cache_dir_honours_the_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONNECTORCH_CACHE", str(tmp_path / "elsewhere"))
    assert default_cache_dir() == tmp_path / "elsewhere"
