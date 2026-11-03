CREATE TABLE subscription_instances (
    -- Hash keys avoid PostgreSQL's btree entry-size limit for percent-encoded URLs.
    instance_key bytea PRIMARY KEY CHECK (octet_length(instance_key) = 32),
    instance text NOT NULL
);

CREATE TABLE subscription_sheets (
    google_sheet_id varchar(128) PRIMARY KEY
        CHECK (google_sheet_id COLLATE "C" ~ '^[A-Za-z0-9_-]{1,128}$'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    next_due timestamptz,
    last_success timestamptz,
    last_failure timestamptz,
    pending_delete boolean NOT NULL DEFAULT false,
    CHECK (NOT pending_delete OR next_due IS NULL)
);

CREATE TABLE sheet_subscriptions (
    instance_key bytea NOT NULL REFERENCES subscription_instances(instance_key),
    google_sheet_id varchar(128) NOT NULL REFERENCES subscription_sheets(google_sheet_id),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    last_submitted_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (instance_key, google_sheet_id)
);
CREATE INDEX sheet_subscriptions_by_sheet ON sheet_subscriptions(google_sheet_id);
CREATE INDEX subscription_sheets_due ON subscription_sheets(next_due)
    WHERE next_due IS NOT NULL AND NOT pending_delete;
CREATE INDEX subscription_sheets_pending_delete ON subscription_sheets(google_sheet_id)
    WHERE pending_delete;

CREATE TABLE subscription_runs (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    google_sheet_id varchar(128) NOT NULL REFERENCES subscription_sheets(google_sheet_id),
    operation text NOT NULL CHECK (operation IN ('refresh', 'delete')),
    status text NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    finished_at timestamptz,
    error_code text,
    CHECK ((status = 'running') = (finished_at IS NULL)),
    CHECK (finished_at IS NULL OR finished_at >= started_at)
);
CREATE UNIQUE INDEX subscription_runs_one_active_per_sheet
    ON subscription_runs(google_sheet_id) WHERE status = 'running';
CREATE INDEX subscription_runs_history
    ON subscription_runs(google_sheet_id, started_at DESC);
