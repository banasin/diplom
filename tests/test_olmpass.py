import pandas as pd
from src.data.olmpass import parse_targets_csv, load_olmpass_iap


def _write_csv(path):
    path.write_text(
        "No;N;IAP;Activity Type;Identifier;Type Target;Target organism\n"
        "1;3;0,9991;ARS-binding factor 1;P14164;Protein;Yeast\n"
        "2;5;1;Ampicillin;CID 6249;LMW compound;\n",
        encoding="utf-8-sig")


def test_parse_targets_csv(tmp_path):
    f = tmp_path / "DNA_MNA_22_targets.csv"
    _write_csv(f)
    df = parse_targets_csv(f)
    assert len(df) == 2
    assert abs(df["IAP"].iloc[0] - 0.9991) < 1e-6
    assert df["Identifier"].iloc[0] == "P14164"


def test_load_olmpass_iap(tmp_path):
    d = tmp_path / "150nM"
    d.mkdir()
    _write_csv(d / "DNA_Ngram_14_targets.csv")
    out = load_olmpass_iap(str(tmp_path))
    assert set(out.columns) >= {"threshold", "aptamer_type", "descriptor",
                                "target_identifier", "iap", "n_actives"}
    assert out["aptamer_type"].iloc[0] == "DNA"
    assert out["descriptor"].iloc[0] == "Ngram"
    assert out["threshold"].iloc[0] == "150nM"
