from src.config import load_config


def test_load_config_reads_expected_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "symbol: SPY\n"
        "horizons_seconds: [0, 1, 2]\n"
    )
    config = load_config(str(config_path))
    assert config["symbol"] == "SPY"
    assert config["horizons_seconds"] == [0, 1, 2]
