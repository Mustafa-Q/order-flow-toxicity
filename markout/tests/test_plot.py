from pathlib import Path
import polars as pl
from src.plot import plot_markout_curve


def test_plot_markout_curve_writes_a_nonempty_png(tmp_path):
    se_table = pl.DataFrame(
        {
            "horizon": [0.1, 1, 10, 100],  # log-x needs h>0; h=0 handled separately in plot.py
            "mean_bps_sw": [1.0, 0.8, 0.5, 0.3],
            "se_bps_sw": [0.1, 0.1, 0.1, 0.1],
            "mean_fracspread_sw": [1.0, 0.9, 0.6, 0.4],
            "se_fracspread_sw": [0.05, 0.05, 0.05, 0.05],
        }
    )
    out_path = tmp_path / "markout_curve.png"
    plot_markout_curve(se_table, sample_period="2026-08-03 to 2026-08-07", n_trades=1234, output_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 1000
