"""Refresh every managed index through the HIA search API, one at a time.

Run by .github/workflows/refresh-indexes.yml. Only the API URL and write key are needed;
backend credentials stay in the app. Logs, summaries and issues show index names only,
never sheet IDs or sheet content.

Environment: HIA_API_URL, API_KEY_WRITE; optional GITHUB_TOKEN, GITHUB_REPOSITORY,
GITHUB_STEP_SUMMARY, HIA_ENVIRONMENT (label used in issue titles).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta

ISSUE_LABEL = "index-refresh"
FAILURE_GRACE = timedelta(days=1)
REFRESH_TIMEOUT_S = 300


def _request(method: str, url: str, headers: dict, body: dict | None = None, timeout: float = 60):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={**headers, "Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload) if payload else None


def _error_detail(error: Exception) -> str:
    if isinstance(error, urllib.error.HTTPError):
        try:
            body = json.loads(error.read() or b"null")
        except ValueError:
            body = None
        if isinstance(body, dict):
            detail = body.get("detail") or (body.get("error") or {}).get("message")
            if isinstance(detail, str):
                return f"{error.code}: {detail}"
        return f"{error.code}: {error.reason}"
    return type(error).__name__


class Api:
    def __init__(self, base_url: str, write_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": write_key}

    def list_indexes(self) -> list[dict]:
        return _request("GET", f"{self.base_url}/vector-stores", self.headers)

    def refresh(self, sheet_id: str) -> str:
        message = _request(
            "POST", f"{self.base_url}/create-vector-store", self.headers,
            {"googleSheetId": sheet_id}, timeout=REFRESH_TIMEOUT_S,
        )
        return "unchanged" if " is unchanged " in str(message) else "updated"


def refresh_all(api: Api) -> list[dict]:
    results = []
    for index in api.list_indexes():
        name = index["index_name"]
        if index["source_type"] != "googlesheet":
            results.append({"index_name": name, "status": "skipped", "detail": "JSON source"})
            continue
        try:
            status, detail = api.refresh(index["source_sheet_id"]), ""
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            status, detail = "failed", _error_detail(error)
            print(f"::warning::Refresh of {name} failed: {detail}")
        results.append({"index_name": name, "status": status, "detail": detail})
    return results


def long_failures(indexes: list[dict], now: datetime) -> dict[str, dict]:
    failing = {}
    for index in indexes:
        since = index.get("failing_since")
        if since and now - datetime.fromisoformat(since) > FAILURE_GRACE:
            failing[index["index_name"]] = index
    return failing


def summary(results: list[dict], environment: str) -> str:
    lines = [f"### Index refresh ({environment})", "", "| Index | Status | Detail |", "|---|---|---|"]
    lines += [f"| `{r['index_name']}` | {r['status']} | {r['detail']} |" for r in results]
    if not results:
        lines.append("| _none_ | | |")
    return "\n".join(lines) + "\n"


class Issues:
    def __init__(self, repository: str, token: str, environment: str):
        self.url = f"https://api.github.com/repos/{repository}/issues"
        self.headers = {
            "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28",
        }
        self.prefix = f"Index refresh failing ({environment}): "

    def sync(self, failing: dict[str, dict], healthy: set[str]) -> None:
        """Open one issue per long-failing index; close it once the index recovers."""
        open_issues = _request(
            "GET", f"{self.url}?labels={ISSUE_LABEL}&state=open&per_page=100", self.headers,
        )
        by_index = {
            issue["title"].removeprefix(self.prefix): issue
            for issue in open_issues
            if issue["title"].startswith(self.prefix) and "pull_request" not in issue
        }
        for name, index in failing.items():
            if name not in by_index:
                _request("POST", self.url, self.headers, {
                    "title": self.prefix + name,
                    "labels": [ISSUE_LABEL],
                    "body": (
                        f"Index `{name}` has failed to refresh since {index['failing_since']}.\n\n"
                        f"Last error: `{index.get('last_error')}`\n\n"
                        "The index is preserved and retried on every scheduled run. This issue closes "
                        "automatically after the next successful refresh."
                    ),
                })
        for name, issue in by_index.items():
            if name in healthy:
                _request("POST", f"{self.url}/{issue['number']}/comments", self.headers, {
                    "body": "Refresh succeeded again; closing.",
                })
                _request("PATCH", f"{self.url}/{issue['number']}", self.headers, {
                    "state": "closed", "state_reason": "completed",
                })


def main() -> int:
    environment = os.environ.get("HIA_ENVIRONMENT", "default")
    api = Api(os.environ["HIA_API_URL"], os.environ["API_KEY_WRITE"])
    results = refresh_all(api)
    text = summary(results, environment)
    print(text)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)

    token, repository = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if token and repository:
        # Re-read metadata so failure start times include this run's failures.
        indexes = api.list_indexes()
        healthy = {i["index_name"] for i in indexes if not i.get("failing_since")}
        Issues(repository, token, environment).sync(
            long_failures(indexes, datetime.now(UTC)), healthy,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
