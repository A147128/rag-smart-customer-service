from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _imports_in_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])

    return imports


def _python_files(package: str) -> list[Path]:
    package_dir = PROJECT_ROOT / package
    return [path for path in package_dir.rglob("*.py") if "__pycache__" not in path.parts]


def test_service_layer_must_not_depend_on_interfaces_or_agent() -> None:
    """service is business logic and must not import upper layers."""
    forbidden = {"api", "ui", "agent"}
    violations: list[str] = []

    for path in _python_files("service"):
        illegal_imports = sorted(_imports_in_file(path) & forbidden)
        if illegal_imports:
            relative_path = path.relative_to(PROJECT_ROOT).as_posix()
            violations.append(f"{relative_path} imports {', '.join(illegal_imports)}")

    assert not violations, "service layer dependency violations:\n" + "\n".join(violations)
