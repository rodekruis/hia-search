# Automated index refresh plan

Steps 1–3 are implemented. Scheduled refresh and cleanup remain planned.

## 1. Index metadata (implemented)

Each successfully created index has one internal document, `hia-index-metadata`, with no content or vector. Azure document keys cannot start with `_`, so `__meta__` is not a valid key. All content counts, full-document reads, and similarity searches exclude this record.

The schema includes:

| Field | Purpose |
|---|---|
| `source_sheet_id` | Original, case-sensitive sheet ID, including for JSON uploads |
| `source_type` | `googlesheet` or `json` |
| `index_owner`, `index_environment` | Scope future jobs to `hia-search` and its `ENVIRONMENT` |
| `created_at` | UTC creation baseline for a never-searched index |
| `pipeline_version`, `embedding_model` | Identify the processing pipeline and configured embedding deployment |
| `sheet_hash` | Hash of the source at the last successful refresh |
| `last_refreshed_at` | Time of the last successful refresh |
| `last_error` | Last refresh failure |
| `last_searched_at` | Usage signal for cleanup |

Successful refreshes set `sheet_hash` and `last_refreshed_at`; failures record `last_error`, which is cleared on recovery. Successful searches update `last_searched_at`. The pipeline version must change when cleaning, translation, or document identity changes. A deployment name alone does not detect a model replaced behind the same name.

There is no legacy mapping or backfill. Delete old indexes manually before rollout; the application does not delete or migrate them automatically. Public-sheet indexes can be recreated on search or chat; JSON-sourced indexes may require another upload.

## 2. Idempotent incremental build (implemented)

- `ensure_index_fresh(index_name, force=False)` is shared by the create endpoint and initial search/chat creation. For a new index, callers also supply `source_sheet_id`; existing indexes identify their own source. Read both relevant public CSV tabs before updating the index, and use the same returned values for hashing and indexing. No Sheets API key is required. The tabs are fetched separately and can reflect different moments if the sheet changes between requests.
- Normalize values deterministically, trimming whitespace and empty trailing rows, then hash the canonical snapshot with SHA-256. Include pipeline-version changes in the freshness decision. When the hash, pipeline, and actual document metadata match, return a result with `"status": "unchanged"` without translating or embedding. `force=true` bypasses the hash check, not document-level reuse.
- Preserve one document per Q&A or offer. IDs are SHA-256 hashes of canonical source content, meaningful metadata, the pipeline version, and a duplicate-occurrence number. This preserves distinct entries with the same text as well as identical duplicate rows. Merge row-number metadata such as `google_index` when rows move, without re-embedding unchanged entries.
- Page through existing IDs **and metadata**, never vectors, excluding the reserved record. Reading metadata rather than IDs alone lets a retry reconcile interrupted row-metadata updates or a reverted source even when the last successful hash matches. Exclude already-seen IDs between pages because equal-score results do not have a stable ordering for offset paging.
- Translate, embed, and upload only new IDs. Translation failures raise rather than committing a fallback embedding. Check each indexing result and retry transient per-document failures; the Azure SDK handles request retries and oversized batches. Wait up to 30 seconds for new IDs to become searchable before deleting obsolete IDs. Check deletions too, and update the hash and refresh timestamp only after the whole update succeeds.
- Register a new index's source before content writes, so interrupted initial uploads remain identifiable. Keep the successful hash unchanged until the final commit, and preserve creation/usage metadata during refresh.
- Validate both responses before deleting anything. Detected read or CSV failures, missing expected headers, and incomplete visible entries preserve existing content. Blank or ID-only placeholder rows and hidden entries are skipped; a valid source with no indexable rows, including a header-only or all-hidden source, clears content documents but retains metadata. The CSV endpoint can silently omit cells or return the first tab for a missing tab, so validation cannot prove that every source value was returned.
- Azure indexing is eventually consistent: searches can temporarily see old and new versions, and the update is not an atomic snapshot. A failed or interrupted run must remain retryable.
- Changing an embedding model or vector dimensions requires a new index and a planned migration, even when a different model has the same dimensions.
- Preserve the create endpoint's JSON-upload behavior and JSON-string response shape. JSON-sourced indexes require another explicit JSON payload for refresh; future scheduled jobs must skip/report them, not replace uploaded content with sheet content.

## 3. Search and usage (implemented)

- The API keeps no memory of indexes: every request derives the index name and queries Azure directly with a hybrid (text + vector) search. A search that finds no index creates it from the public sheet with `ensure_index_fresh` and retries once; if the sheet cannot be read, the refresh error is returned. A deleted JSON-sourced index is therefore not recoverable by search and needs another upload.
- Two simultaneous first searches of a missing index may both build it. Content-addressed IDs keep the result correct; only the embedding cost is paid twice.
- An existing but empty or partially built index is not repaired by search; the next create call or scheduled refresh reconciles it.
- Every successful search updates `last_searched_at` with a small metadata merge. A failed usage write is logged and does not fail the search.
- Index names use hashes of the exact sheet ID and environment, consistently for creation, lookup, and deletion. Delete indexes created with the old naming scheme manually.

## 4. Scheduled refresh workflow (implemented)

- `.github/workflows/refresh-indexes.yml` runs every 6 hours (and on demand) for the `production` GitHub environment only; dev indexes are refreshed on demand or when recreated by search. It runs `scripts/refresh_indexes.py`, which only calls the API. GitHub holds the production API URL (`HIA_API_URL` variable) and write key (`API_KEY_WRITE` secret), not Azure, OpenAI or translator credentials.
- `GET /vector-stores` (write key) lists the indexes owned by this application in the current environment, with their metadata. Indexes without recognized metadata are skipped. The script then calls `POST /create-vector-store` for each sheet-backed index, one at a time, so each refresh stays a normal, bounded request. JSON-sourced indexes are skipped.
- On a failure the API records `last_error` as `"<failing since, UTC ISO> | <message>"`. The start time is kept across repeated failures and cleared by the next successful refresh. The index is preserved and the script continues with the next one.
- The run summary lists every index's status. An index failing for more than a day gets one issue labelled `index-refresh`; the issue is closed automatically once that index refreshes successfully. Summaries and issues show index names only, never sheet IDs or contents.
- The job fails only when indexes cannot be listed. Required protection rules on the `production` environment would block scheduled runs.
## 5. Daily cleanup (implemented)

- `.github/workflows/refresh-indexes.yml` runs cleanup daily at 03:47 UTC for production. The cleanup job can also be requested manually with the `run_cleanup` workflow input.
- An index is eligible when its most recent `last_searched_at` is more than six calendar months old. If it has never been searched, `created_at` is the baseline. The cutoff is calculated in UTC and clamps to the last day of the target month when needed.
- Only indexes returned by the managed-index list are considered. Missing or invalid activity timestamps are kept. Before deleting each candidate, the script re-lists metadata and keeps it if it was searched since the initial listing.
- The script uses the existing write-key-protected list and delete endpoints. It never logs sheet IDs or contents. Deleting a sheet-backed index allows search to recreate it; a deleted JSON-sourced index must be uploaded again.

## Sheet content conventions

Only the values returned by the public CSV export are indexed. Styling, notes, and hyperlink targets are not exported. Use the existing `#VISIBLE` column to hide entries rather than relying on Google Sheets row visibility or filters; use explicit status columns only when the loader is configured to filter them.

CSV contains displayed values: number or date formatting can change the hash, while rounding can conceal changes to underlying values. Store identifiers such as phone numbers as plain text. The Google visualization endpoint may omit mixed-type cells, and a missing named tab may fall back to the first tab; expected-header checks cannot catch every such case. Avoid volatile formulas or exclude their columns.
