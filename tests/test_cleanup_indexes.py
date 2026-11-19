from __future__ import annotations

from datetime import UTC, datetime

from scripts.cleanup_indexes import cleanup, parse_timestamp, subtract_calendar_months


class FakeApi:
    def __init__(self, indexes, refreshed_before_recheck=()):
        self.indexes = {index["index_name"]: dict(index) for index in indexes}
        self.refreshed_before_recheck = set(refreshed_before_recheck)
        self.list_calls = 0
        self.deleted = []

    def list_indexes(self):
        self.list_calls += 1
        if self.list_calls > 1:
            for name in self.refreshed_before_recheck:
                self.indexes[name]["last_searched_at"] = "2025-11-20T00:00:00Z"
        return [dict(index) for index in self.indexes.values()]

    def delete(self, sheet_id):
        self.deleted.append(sheet_id)


def _index(name, *, searched=None, created="2025-05-01T00:00:00Z", source="googlesheet"):
    return {
        "index_name": name,
        "source_sheet_id": f"sheet-{name}",
        "source_type": source,
        "created_at": created,
        "last_searched_at": searched,
    }


def test_six_calendar_month_cutoff_clamps_to_last_day_of_target_month():
    assert subtract_calendar_months(datetime(2024, 8, 31, 12, tzinfo=UTC), 6) == datetime(
        2024, 2, 29, 12, tzinfo=UTC,
    )
    assert subtract_calendar_months(datetime(2025, 8, 31, 12, tzinfo=UTC), 6) == datetime(
        2025, 2, 28, 12, tzinfo=UTC,
    )


def test_cleanup_uses_search_or_creation_time_and_keeps_recent_or_missing_dates():
    api = FakeApi([
        _index("stale-search", searched="2025-05-01T00:00:00Z"),
        _index("stale-never-searched", searched=None, created="2025-05-01T00:00:00Z"),
        _index("recent", searched="2025-11-19T00:00:00Z"),
        _index("missing-date", searched=None, created=None),
        _index("old-json", searched="2025-05-01T00:00:00Z", source="json"),
    ])

    results = cleanup(api, datetime(2025, 11, 19, 0, 0, 1, tzinfo=UTC))

    assert api.deleted == ["sheet-stale-search", "sheet-stale-never-searched", "sheet-old-json"]
    assert [(result["index_name"], result["status"]) for result in results] == [
        ("stale-search", "deleted"),
        ("stale-never-searched", "deleted"),
        ("recent", "kept"),
        ("missing-date", "kept"),
        ("old-json", "deleted"),
    ]


def test_cleanup_rechecks_usage_before_deleting_and_preserves_a_recently_used_index():
    api = FakeApi(
        [_index("used-again", searched="2025-05-01T00:00:00Z")],
        refreshed_before_recheck={"used-again"},
    )

    result = cleanup(api, datetime(2025, 11, 19, tzinfo=UTC))

    assert api.deleted == []
    assert result == [{
        "index_name": "used-again",
        "status": "kept",
        "detail": "used since initial listing",
    }]


def test_invalid_or_naive_timestamps_are_not_treated_as_activity():
    assert parse_timestamp(None) is None
    assert parse_timestamp("not-a-date") is None
    assert parse_timestamp("2025-01-01T00:00:00") is None
    assert parse_timestamp("2025-01-01T00:00:00Z") == datetime(2025, 1, 1, tzinfo=UTC)
