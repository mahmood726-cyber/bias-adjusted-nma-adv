"""Preflight Stan/NUTS reference matching without treating skips as passes."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from pathlib import Path


DEFAULT_MODEL = Path("external/stan/standard_binary_nma.stan")
DEFAULT_BACKEND = Path("src/bias_nma_adv/stan_backend.py")
DEFAULT_SCRIPT = Path("scripts/preflight_stan_nuts_adapter.py")
DEFAULT_OUTPUT = Path("validation/reference_runs/stan_nuts_cmdstan_preflight.toml")
DEFAULT_REFERENCE_OUTPUT = Path("validation/reference_runs/stan_nuts_cmdstan_output.json")
TEXT_HASH_EXTENSIONS = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".r",
    ".stan",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256_file(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in TEXT_HASH_EXTENSIONS:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def toml_path(path: Path) -> str:
    return path.as_posix()


def build_report(root: Path, checked_at: str) -> str:
    from bias_nma_adv.stan_backend import cmdstan_preflight

    input_paths = [DEFAULT_MODEL, DEFAULT_BACKEND, DEFAULT_SCRIPT]
    for relative_path in input_paths:
        if not (root / relative_path).is_file():
            raise FileNotFoundError(relative_path)

    preflight = cmdstan_preflight()
    if preflight.status == "available":
        status = "failed"
        skip_reason = (
            "CmdStan is available, but this preflight does not execute NUTS. "
            "Run the Stan model, export diagnostics, and validate output before certification."
        )
    else:
        status = "unavailable"
        skip_reason = preflight.message

    package_versions: dict[str, str] = {}
    if preflight.cmdstanpy_version:
        package_versions["cmdstanpy"] = preflight.cmdstanpy_version
    if preflight.cmdstan_available:
        try:
            import cmdstanpy  # type: ignore[import-not-found]

            cmdstan_version = cmdstanpy.cmdstan_version()
            package_versions["cmdstan"] = ".".join(str(part) for part in cmdstan_version)
        except Exception:
            package_versions["cmdstan"] = "available_version_unresolved"

    command = [
        "python",
        toml_path(DEFAULT_SCRIPT),
        "--root",
        ".",
        "--output",
        toml_path(DEFAULT_OUTPUT),
    ]

    lines = [
        'schema_version = "reference_run/v1"',
        'target_id = "bayesian_nma_multinma_cmdstan"',
        'adapter_id = "python_cmdstan_nuts_preflight"',
        'reference_method = "CmdStanPy/CmdStan NUTS"',
        f'status = "{status}"',
        'certification_effect = "none"',
        f'checked_at = "{checked_at}"',
        "command = [" + ", ".join(f'"{part}"' for part in command) + "]",
        'executable = "python"',
        f"executable_found = true",
        "package_versions = {"
        + ", ".join(f'{key} = "{value}"' for key, value in sorted(package_versions.items()))
        + "}",
        "input_artifacts = [",
        *[f'  "{toml_path(path)}",' for path in input_paths],
        "]",
        "output_artifacts = []",
        'tolerance = ""',
        f'skip_reason = "{skip_reason}"',
        'required_diagnostics = ["r_hat", "ess_bulk", "ess_tail", "divergent_transitions", "treedepth_saturation", "mcse", "prior_predictive_checks", "posterior_predictive_checks"]',
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
