from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_contract_distinguishes_barrier_and_conductivity() -> None:
    text = (ROOT / "RESEARCH_CONTRACT.md").read_text(encoding="utf-8").lower()
    assert "migration barrier is equivalent to ionic conductivity" in text
    assert "forbidden claims" in text


def test_new_dft_is_forbidden() -> None:
    text = (ROOT / "RESEARCH_CONTRACT.md").read_text(encoding="utf-8").lower()
    assert "new dft/neb/aimd/md" in text or "new dft, neb, aimd" in text
