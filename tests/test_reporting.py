from src.reporting.report import generate_report


def test_report_generation(tmp_path):
    path = generate_report(tmp_path / "report.md", "P1", "DEMO", {"MAPK": 1.0}, [])
    text = path.read_text()
    assert "RESEARCH USE ONLY" in text
    assert "SYNTHETIC DATA" in text
