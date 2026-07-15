import csv
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from bias_nma_adv.multiarm import ContrastRow, fit_multiarm_gls


ROOT = Path(__file__).resolve().parents[1]
ARMS_PATH = ROOT / "validation" / "multiarm" / "netmeta_portfolio_multiarm_arms.csv"
ARTIFACT_PATH = ROOT / "validation" / "multiarm" / "netmeta_portfolio_multiarm_benchmark.toml"
WRITER_SCRIPT = ROOT / "scripts" / "write_multiarm_benchmark.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arm_rows() -> tuple[dict[str, object], ...]:
    with ARMS_PATH.open("r", encoding="utf-8", newline="") as handle:
        return tuple(
            {
                "fixture_id": row["fixture_id"],
                "study": row["study"],
                "treatment": row["treatment"],
                "events": int(row["events"]),
                "n": int(row["n"]),
            }
            for row in csv.DictReader(handle)
        )


def log_odds(events: int, n: int) -> float:
    return math.log(events / (n - events))


def contrast_rows_from_arms(arms: tuple[dict[str, object], ...]) -> tuple[ContrastRow, ...]:
    by_study: dict[str, list[dict[str, object]]] = {}
    for arm in arms:
        by_study.setdefault(str(arm["study"]), []).append(arm)

    rows: list[ContrastRow] = []
    for study, study_arms in by_study.items():
        for i in range(len(study_arms)):
            for j in range(i + 1, len(study_arms)):
                arm_1 = study_arms[i]
                arm_2 = study_arms[j]
                e1 = int(arm_1["events"])
                n1 = int(arm_1["n"])
                e2 = int(arm_2["events"])
                n2 = int(arm_2["n"])
                rows.append(
                    ContrastRow(
                        study=study,
                        t1=str(arm_1["treatment"]),
                        t2=str(arm_2["treatment"]),
                        est=log_odds(e2, n2) - log_odds(e1, n1),
                        se=math.sqrt(1 / e1 + 1 / (n1 - e1) + 1 / e2 + 1 / (n2 - e2)),
                    )
                )
    return tuple(rows)


def test_multiarm_benchmark_artifact_recomputes_from_arm_fixture():
    with ARTIFACT_PATH.open("rb") as handle:
        artifact = tomllib.load(handle)

    assert artifact["schema_version"] == "multiarm_benchmark/v1"
    assert artifact["status"] == "local_fixture_recomputed"
    assert artifact["certification_effect"] == "none"
    assert artifact["source_policy"] == "portfolio_algorithmic_fixture_not_clinical_evidence"
    assert artifact["reference_target_id"] == "multiarm_gls_netmeta_portfolio_fixture"
    assert artifact["arm_data"] == "validation/multiarm/netmeta_portfolio_multiarm_arms.csv"
    assert artifact["arm_data_sha256"] == sha256_file(ARMS_PATH)

    arm_rows = load_arm_rows()
    by_fixture = {
        fixture["fixture_id"]: tuple(
            row for row in arm_rows if row["fixture_id"] == fixture["fixture_id"]
        )
        for fixture in artifact["fixtures"]
    }
    effect_rows = {
        (effect["fixture_id"], effect["model"], effect["treatment"]): effect
        for effect in artifact["effects"]
    }

    assert set(by_fixture) == {"consistent", "heterogeneous"}
    for fit_row in artifact["model_fits"]:
        key = (fit_row["fixture_id"], fit_row["model"])
        fit = fit_multiarm_gls(
            contrast_rows_from_arms(by_fixture[fit_row["fixture_id"]]),
            reference_treatment=fit_row["reference_treatment"],
            model=fit_row["model"],
        )
        assert fit.tau2 == pytest.approx(fit_row["tau2"], abs=1e-12)
        assert list(fit.multi_arm_studies) == fit_row["multi_arm_studies"]
        assert list(fit.warnings) == fit_row["warnings"]
        if fit.q is not None:
            assert fit.q == pytest.approx(fit_row["q"], abs=1e-10)
        else:
            assert "q" not in fit_row
        if fit.df is not None:
            assert fit.df == fit_row["df"]
        else:
            assert "df" not in fit_row

        for treatment in fit.nonreference_treatments:
            effect = effect_rows[(key[0], key[1], treatment)]
            estimate, se = fit.effect_vs_reference(treatment)
            assert effect["estimate"] == pytest.approx(estimate, abs=1e-12)
            assert effect["se"] == pytest.approx(se, abs=1e-12)


def test_multiarm_benchmark_writer_regenerates_same_artifact(tmp_path):
    output = tmp_path / "multiarm.toml"
    completed = subprocess.run(
        [sys.executable, str(WRITER_SCRIPT), "--output", str(output)],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr

    with ARTIFACT_PATH.open("rb") as handle:
        expected = tomllib.load(handle)
    with output.open("rb") as handle:
        regenerated = tomllib.load(handle)
    assert regenerated == expected
