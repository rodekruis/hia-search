"""Tests for the scheduled refresh script (scripts/refresh_indexes.py)."""

from __future__ import annotations

import io
import urllib.error
from datetime import UTC, datetime

from scripts import refresh_indexes as script


class FakeApi:
    def __init__(self, indexes, failures=()):
        self.indexes, self.failures, self.refreshed = indexes, set(failures), []

    def list_indexes(self):
        return self.indexes

    def refresh(self, sheet_id):
        self.refreshed.append(sheet_id)
        if sheet_id in self.failures:
            raise urllib.error.HTTPError(
                "url", 502, "Bad Gateway", {}, io.BytesIO(b'{"detail": "Could not read sheet"}'),
            )
        return "unchanged"


def _index(name, source_type="googlesheet", failing_since=None):
    return {
        "index_name": name, "source_sheet_id": f"sheet-{name}", "source_type": source_type,
        "failing_since": failing_since, "last_error": "x" if failing_since else None,
    }


def test_refresh_continues_after_failures_and_skips_json_sources():
    api = FakeApi([_index("a"), _index("b"), _index("c", "json")], failures={"sheet-a"})

    results = script.refresh_all(api)

    assert api.refreshed == ["sheet-a", "sheet-b"]
    assert [(r["index_name"], r["status"]) for r in results] == [
        ("a", "failed"), ("b", "unchanged"), ("c", "skipped"),
    ]
    assert results[0]["detail"] == "502: Could not read sheet"
    assert "sheet-" not in script.summary(results, "dev")


def test_only_failures_older_than_a_day_are_reported():
    now = datetime(2025, 1, 3, tzinfo=UTC)
    indexes = [
        _index("old", failing_since="2025-01-01T23:00:00Z"),
        _index("new", failing_since="2025-01-02T12:00:00Z"),
        _index("ok"),
    ]
    assert list(script.long_failures(indexes, now)) == ["old"]


def test_issues_are_deduplicated_and_closed_after_recovery(monkeypatch):
    calls = []
    prefix = "Index refresh failing (dev): "
    open_issues = [
        {"number": 1, "title": prefix + "old"},
        {"number": 2, "title": prefix + "recovered"},
        {"number": 3, "title": prefix + "still-failing-recently"},
    ]

    def fake_request(method, url, headers, body=None, timeout=60):
        calls.append((method, url.rsplit("/issues", 1)[1], body))
        return open_issues if method == "GET" else {}

    monkeypatch.setattr(script, "_request", fake_request)
    failing = {"old": _index("old", failing_since="t"), "fresh": _index("fresh", failing_since="t")}

    script.Issues("org/repo", "token", "dev").sync(failing, healthy={"recovered", "ok"})

    writes = [(method, path, (body or {}).get("title") or (body or {}).get("state"))
              for method, path, body in calls if method != "GET"]
    assert writes == [
        ("POST", "", prefix + "fresh"),
        ("POST", "/2/comments", None),
        ("PATCH", "/2", "closed"),
    ]
