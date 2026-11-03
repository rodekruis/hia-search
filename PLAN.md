# Plan: Subscription-Based Refresh

Add subscriptions that keep vector stores updated while at least one instance needs them.

## API

- Submit `instance` and `googleSheetIds`, the complete current set of sheet IDs for that client. The subscribe route is public and requires no API key or other authentication.
- Record submission timestamps without postponing an existing refresh.
- Allow multiple instances per sheet and multiple sheets per instance.
- Reconcile the submitted set atomically for that instance: add new subscriptions, retain matching subscriptions, and internally remove only previously stored subscriptions omitted from the new set. Leave other instances' subscriptions unchanged.
- An empty `googleSheetIds` list clears that instance's existing subscriptions. Repeated identical submissions are idempotent; removing a relationship that never existed has no effect and must not queue index deletion.
- Internal removal runs as part of successful subscription reconciliation, without calling an HTTP unsubscribe route. Subscriptions do not expire merely because a client stops submitting.
- Also provide an explicit unsubscribe route protected by the existing `API_KEY_WRITE` / `require_write_key` mechanism. Accept an `instance` and `googleSheetId` pair and remove only that previously stored relationship, idempotently. Reuse the same internal removal logic as reconciliation; never expose an unauthenticated unsubscribe route.
- When the last instance is removed from a previously subscribed sheet, stop refreshes and queue index deletion. Retain the index and continue scheduled refreshes while any instance remains subscribed.

### Input Validation

- Require `instance` to be a string and `googleSheetIds` to be an explicitly supplied list of strings. Reject a missing list rather than interpreting it as an empty set. Validate the entire request before any changes; return HTTP 422 for invalid input without adding or removing subscriptions or scheduling work. Deduplicate valid sheet IDs when comparing sets.
- Each entry in `googleSheetIds` must be a bare identifier containing only ASCII letters, digits, underscores, and hyphens, with a service-defined limit of 1 to 128 characters. Reject full spreadsheet URLs, whitespace, slashes, query strings, and other characters rather than extracting or repairing an ID. This is an API input policy, not a guarantee of Google's identifier length or that a sheet exists or is accessible.
- `instance` must be an absolute HTTPS URL with a valid hostname. Use Pydantic's `HttpUrl` validation and its built-in 2,083-character maximum rather than implementing a custom length rule or URL regex; restrict the allowed scheme to HTTPS. Reject HTTP, relative URLs, other schemes, embedded credentials, and whitespace. Validate the entire submitted URL before stripping query parameters or fragments. Allow any path depth within the built-in limit. This limit is chosen to minimize custom validation code, not as a URL-standard requirement. Infrastructure request-size limits still apply.
- Normalize instance URLs consistently before subscription lookup or storage. Identity consists of the HTTPS protocol, hostname (and any non-default port), and full path. Lowercase the scheme and hostname, remove the default HTTPS port (443), and normalize an absent path to `/`. Preserve path casing, segments, and trailing slashes rather than truncating to a fixed number of segments.
- Strip query parameters and fragment identifiers during normalization; neither is stored as part of the instance identifier or used for subscription identity. For example, `https://example.org/client/site?lang=en#section` and `https://example.org/client/site` identify the same instance. Equivalent normalized URLs must reconcile the same instance's subscriptions.
- Treat the instance URL only as an identifier: do not fetch it. URL validity does not prove that the caller owns or controls that instance.
- Apply the same sheet-ID validation and instance-URL normalization to explicit unsubscribe requests.

## Storage

Use an entirely separate PostgreSQL server with dedicated credentials, migrations, and tables for subscriptions, sheet scheduling state, and run history. No coupling to conversation storage.

### Data-model

At least the following information should be retreivable/usable for scheduling and subscription management.

- `googleSheetId`: Identifier for the Google Sheet.
- `created_at`: Timestamp of subscription.
- `instances`: Validated, normalized URLs for the subscribing instances. (Array of instance identifiers)
- Store unique `(instance, googleSheetId)` relationships with per-relationship `created_at` and `last_submitted_at` timestamps so reconciliation can remove only that instance's previous subscriptions. The `instances` array can be derived from these relationships.
- `next_due`: Timestamp of the next scheduled refresh.
- `last_success`: Timestamp of the last successful refresh.
- `last_failure`: Timestamp of the last failed refresh.

## Scheduling

- Check for work approximately hourly.
- First refresh runs at the next check.
- Subsequent refreshes become due 12 hours after successful completion.
- Failures retry after an hour and remain recorded.
- Dispatch refreshes only for sheets with at least one current subscription. Coordinate reconciliation, refresh claims, and pending deletion to prevent concurrent submissions or dispatchers from deleting a still-subscribed sheet. A new subscription before deletion starts cancels pending deletion; if deletion is already running, serialize it with an immediately due recreation.
- Serialize refresh and deletion per sheet. Let an active rebuild finish before deletion.
- Keep scheduler hosting undecided; provide a standalone dispatcher command.

## Implementation

1. Define schemas with strict sheet-ID and instance-URL validation, configuration, and database migrations.
2. Implement atomic per-instance subscription-set reconciliation, internal removal of omitted existing subscriptions, and due-time logic.
3. Reuse `create_vector_store_index` for creation and refresh.
4. Add the public subscription-set route, write-key-protected unsubscribe route, and dispatcher. Reuse existing API-key authentication without introducing per-client key management.
5. Update packaging and deployment documentation.

## Verification

Test subscription deduplication, timing boundaries, retries, concurrent dispatchers, last-unsubscribe deletion, resubscription races, restart recovery, and existing endpoint compatibility.

Verify subscription-set submission succeeds without credentials. Verify explicit unsubscribe succeeds with the write key and rejects missing, incorrect, or read-only keys without changing stored state. Test that explicit unsubscribe removes only the requested existing pair, preserves other subscribers, and does not queue deletion for unknown relationships or repeated removals.

Test replacing `[A, B]` with `[B, C]` for one instance removes only its relationship with A, preserves B's scheduling state, and adds C. If another instance still subscribes to A, A must continue refreshing and must not be deleted. Test an empty list, duplicate IDs, identical repeated submissions, unknown instances submitting empty lists, concurrent set replacements, and removal only of previously stored relationships.

Test sheet-ID character and length boundaries, rejection of full sheet URLs, missing or incorrectly typed lists, invalid individual entries, URL scheme/hostname checks, credential/whitespace rejection, and consistent URL normalization. Cover rejection of HTTP and other non-HTTPS schemes, acceptance at the built-in 2,083-character limit, and rejection above that limit (including URLs whose excess length is in queries or fragments). Cover deep paths within the limit, path differences remaining distinct, hostname casing and default-HTTPS-port equivalence, empty-path normalization, and query/fragment variations resolving to the same stored instance across subscription and explicit unsubscribe requests. Invalid requests must neither partially reconcile subscriptions nor schedule work. Confirm the instance URL is never fetched.

## Important Caveats

Existing rebuilds delete and recreate indexes, so updates are not atomic. Coordination with existing manual mutation endpoints and cross-process cached clients needs verification before rollout.

Public subscriptions can trigger recurring translation and embedding costs and can impersonate another instance URL. Strict input validation does not prevent this abuse. Public subscriptions can also prevent the last-subscriber cleanup by registering additional instances.

Because subscription-set replacement is unauthenticated, a caller who supplies another instance's URL can replace that instance's subscriptions, including clearing them with an empty list. Protecting the explicit unsubscribe route does not prevent this public-reconciliation path.

These risks are explicitly accepted for the current project scope. Per-client API-key management or another instance-ownership verification mechanism is deferred because it is not currently available. The shared write key protects explicit unsubscribe but does not establish per-instance ownership.

## Future Abuse Mitigations

Rate limits, subscription quotas, cost monitoring, and instance-ownership verification are future improvements, not prerequisites for this version. Keep request handling, reconciliation, and authentication separate so these controls can be added later without changing the subscription data model or scheduling rules. Specific mechanisms and thresholds remain undecided.
