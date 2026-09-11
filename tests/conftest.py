"""Shared pytest configuration."""


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--regenerate-golden",
        action="store_true",
        default=False,
        help="rewrite the regression fixtures instead of comparing against them",
    )
