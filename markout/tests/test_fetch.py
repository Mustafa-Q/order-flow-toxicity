from datetime import date
from unittest.mock import MagicMock
from src.fetch import compute_trading_days, resolve_dataset


def test_compute_trading_days_excludes_weekends_and_today():
    # 2026-08-13 is a Thursday
    days = compute_trading_days(5, as_of=date(2026, 8, 13))
    assert days == [
        date(2026, 8, 6),
        date(2026, 8, 7),
        date(2026, 8, 10),
        date(2026, 8, 11),
        date(2026, 8, 12),
    ]
    assert date(2026, 8, 13) not in days  # today is never "complete"
    assert all(d.weekday() < 5 for d in days)  # no weekends


def test_resolve_dataset_prefers_consolidated_feed():
    client = MagicMock()
    client.metadata.list_datasets.return_value = ["XNAS.ITCH", "EQUS.MINI"]
    dataset = resolve_dataset(client, {"dataset_override": None})
    assert dataset == "EQUS.MINI"


def test_resolve_dataset_respects_override():
    client = MagicMock()
    dataset = resolve_dataset(client, {"dataset_override": "XNYS.PILLAR"})
    assert dataset == "XNYS.PILLAR"
    client.metadata.list_datasets.assert_not_called()


def test_parse_as_of_returns_date():
    from src.fetch import parse_as_of

    assert parse_as_of("2026-07-21") == date(2026, 7, 21)
    assert parse_as_of(None) is None


def test_compute_trading_days_anchored_window_contains_cached_days():
    # anchoring the 20-day window at 2026-07-21 must produce a window whose
    # last two days are the two real days already cached on disk
    days = compute_trading_days(20, as_of=date(2026, 7, 21))
    assert days[-1] == date(2026, 7, 20)
    assert days[-2] == date(2026, 7, 17)
    assert len(days) == 20
