"""Delete managed production indexes inactive for at least six calendar months."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime


def subtract_calendar_months(value: datetime, months: int) -> datetime:
    month_index = value.year * 12 + value.month - 1 - months
    year, month = divmod(month_index, 12)
    month += 1
    # Clamp to the last day of the target month.
    import calendar

    return value.replace(year=year, month=month, day=min(value.day, calendar.monthrange(year, month)[1]))


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return None
    return timestamp.astimezone(UTC)


def _request(method: str, url: str, headers: dict, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={**headers, "Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    return json.loads(payload) if payload else None


class Api:
    def __init__(self, base_url: str, write_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": write_key}

    def list_indexes(self) -> list[dict]:
        return _request("GET", f"{self.base_url}/vector-stores", self.headers)

    def delete(self, sheet_id: str) -> None:
        _request(
            "DELETE", f"{self.base_url}/delete-vector-store", self.headers,
            {"googleSheetId": sheet_id},
        )


def _activity(index: dict) -> datetime | None:
    return parse_timestamp(index.get("last_searched_at")) or parse_timestamp(index.get("created_at"))


def cleanup(api: Api, now: datetime) -> list[dict]:
    """Delete only indexes still inactive at the moment each deletion is attempted."""
    cutoff = subtract_calendar_months(now.astimezone(UTC), 6)
    initial_indexes = api.list_indexes()
    results = []

    for candidate in initial_indexes:
        name = candidate["index_name"]
        activity = _activity(candidate)
        if activity is None or activity >= cutoff:
            results.append({"index_name": name, "status": "kept", "detail": "recent or missing activity"})
            continue

        # Re-read usage just before deleting so a newly used index is retained.
        current = next(
            (index for index in api.list_indexes() if index["index_name"] == name),
            None,
        )
        if current is None:
            results.append({"index_name": name, "status": "skipped", "detail": "no longer listed"})
            continue
        current_activity = _activity(current)
        if current_activity is None or current_activity >= cutoff:
            results.append({"index_name": name, "status": "kept", "detail": "used since initial listing"})
            continue

        try:
            api.delete(current["source_sheet_id"])
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            status = f"HTTP {error.code}" if isinstance(error, urllib.error.HTTPError) else type(error).__name__
            results.append({"index_name": name, "status": "failed", "detail": status})
            print(f"::warning::Could not delete {name}: {status}")
        else:
            results.append({"index_name": name, "status": "deleted", "detail": ""})
    return results


def summary(results: list[dict], environment: str) -> str:
    lines = [
        f"### Index cleanup ({environment})", "",
        "| Index | Status | Detail |", "|---|---|---|",
    ]
    lines.extend(
        f"| `{result['index_name']}` | {result['status']} | {result['detail']} |"
        for result in results
    )
    if not results:
        lines.append("| _none_ | | |")
    return "\n".join(lines) + "\n"


def main() -> int:
    environment = os.environ.get("HIA_ENVIRONMENT", "production")
    api = Api(os.environ["HIA_API_URL"], os.environ["API_KEY_WRITE"])
    results = cleanup(api, datetime.now(UTC))
    text = summary(results, environment)
    print(text)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)
    return int(any(result["status"] == "failed" for result in results))


if __name__ == "__main__":
    sys.exit(main())
