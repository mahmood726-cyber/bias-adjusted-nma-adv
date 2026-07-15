"""Preflight the optional netmeta multi-arm fixture adapter without certifying skips."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path
import shutil
import subprocess


DEFAULT_ARMS = Path("validation/multiarm/netmeta_portfolio_multiarm_arms.csv")
DEFAULT_BENCHMARK = Path("validation/multiarm/netmeta_portfolio_multiarm_benchmark.toml")
DEFAULT_ADAPTER = Path("external/r/multiarm_netmeta_fixture.R")
DEFAULT_OUTPUT = Path("validation/reference_runs/multiarm_netmeta_preflight.toml")
DEFAULT_REFERENCE_OUTPUT = Path("validation/reference_runs/multiarm_netmeta_output.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def r_package_version(rscript: str, package_name: str) -> str | None:
    expr = (
        f"if (requireNamespace('{package_name}', quietly=TRUE)) "
        f"cat(as.character(utils::packageVersion('{package_name}'))) else quit(status=42)"
    )
    completed = subprocess.run(
        [rscript, "--vanilla", "-e", expr],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def toml_path(path: Path) -> str:
    return path.as_posix()


def quote_toml(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def find_rscript() -> str | None:
    """Find Rscript from PATH or standard Windows Program Files R installs."""

    path_hit = shutil.which("Rscript")
    if path_hit:
        return path_hit
    candidates: list[Path] = []
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        program_root = os.environ.get(env_name)
        if not program_root:
            continue
        r_root = Path(program_root) / "R"
        if r_root.is_dir():
            candidates.extend(sorted(r_root.glob("R-*\\bin\\Rscript.exe"), reverse=True))
            candidates.extend(sorted(r_root.glob("R-*\\bin\\x64\\Rscript.exe"), reverse=True))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def build_report(root: Path, checked_at: str) -> str:
    rscript = find_rscript()
    input_paths = [
        DEFAULT_ARMS,
        DEFAULT_BENCHMARK,
        DEFAULT_ADAPTER,
    ]
    for relative_path in input_paths:
        if not (root / relative_path).is_file():
            raise FileNotFoundError(relative_path)

    package_versions: dict[str, str] = {}
    executable_found = rscript is not None
    status = "unavailable"
    skip_reason = "Rscript is not available on PATH in this environment; no netmeta multi-arm parity was run."
    if rscript is not None:
        missing: list[str] = []
        r_version = subprocess.run(
            [rscript, "--vanilla", "-e", "cat(as.character(getRversion()))"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r_version.returncode == 0 and r_version.stdout.strip():
            package_versions["R"] = r_version.stdout.strip()
        for package_name in ("netmeta", "meta", "jsonlite"):
            version = r_package_version(rscript, package_name)
            if version is None:
                missing.append(package_name)
            else:
                package_versions[package_name] = version
        if missing:
            skip_reason = (
                "Rscript is available, but required R packages are missing: "
                + ", ".join(sorted(missing))
                + ". No netmeta multi-arm parity was run."
            )
        else:
            status = "failed"
            skip_reason = (
                "Reference adapter prerequisites are available, but this preflight does not execute "
                "the parity adapter. Run the command and validate output before certification."
            )

    command = [
        "Rscript",
        "--vanilla",
        toml_path(DEFAULT_ADAPTER),
        "--arms",
        toml_path(DEFAULT_ARMS),
        "--output",
        toml_path(DEFAULT_REFERENCE_OUTPUT),
    ]

    lines = [
        'schema_version = "reference_run/v1"',
        'target_id = "multiarm_gls_netmeta_portfolio_fixture"',
        'adapter_id = "r_netmeta_multiarm_preflight"',
        'reference_method = "netmeta"',
        f'status = "{status}"',
        'certification_effect = "none"',
        f'checked_at = "{checked_at}"',
        "command = [" + ", ".join(quote_toml(part) for part in command) + "]",
        'executable = "Rscript"',
        f"executable_found = {str(executable_found).lower()}",
        "package_versions = {"
        + ", ".join(f'{key} = "{value}"' for key, value in sorted(package_versions.items()))
        + "}",
        "input_artifacts = [",
        *[f'  "{toml_path(path)}",' for path in input_paths],
        "]",
        "output_artifacts = []",
        'tolerance = ""',
        f"skip_reason = {quote_toml(skip_reason)}",
        "",
        "[input_sha256]",
    ]
    for relative_path in input_paths:
        normalized = toml_path(relative_path)
        lines.append(f'"{normalized}" = "{sha256_file(root / relative_path)}"')
    lines.extend(["", "[output_sha256]", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checked-at", default=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    args = parser.parse_args()

    root = args.root.resolve()
    text = build_report(root, args.checked_at)
    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
