from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_contracts_exist() -> None:
    for name in [
        "RESEARCH_CONTRACT.md",
        "DATA_CONTRACT.md",
        "EXPERIMENT_PROTOCOL.md",
        "DEFINITION_OF_DONE.md",
    ]:
        assert (ROOT / name).exists()


def test_no_atomistic_runtime_dependency_declared() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    for forbidden in ["quantum-espresso", "vasp", "lammps", "cp2k", "gpaw"]:
        assert forbidden not in text
