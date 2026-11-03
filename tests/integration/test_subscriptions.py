from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier
from time import monotonic, sleep

import psycopg
import pytest

from subscriptions.migrate import migrate
from subscriptions.store import SubscriptionStore, open_subscription_store

pytestmark = pytest.mark.integration
INSTANCE = "https://example.org/client"
OTHER = "https://example.org/other"


def relationships(store):
    with store.pool.connection() as conn:
        return set(
            conn.execute(
                """SELECT i.instance, s.google_sheet_id FROM sheet_subscriptions s
               JOIN subscription_instances i USING (instance_key)"""
            )
        )


def sheet(store, sheet_id):
    with store.pool.connection() as conn:
        return conn.execute(
            """SELECT next_due, pending_delete, last_success, last_failure, created_at
               FROM subscription_sheets WHERE google_sheet_id = %s""",
            (sheet_id,),
        ).fetchone()


def test_reconcile_shared_sheets_and_due_time(store):
    before = datetime.now(UTC)
    store.reconcile(INSTANCE, {"A", "B"})
    assert before <= sheet(store, "A")[0] <= datetime.now(UTC)
    store.reconcile(OTHER, {"A"})
    due = datetime.now(UTC) + timedelta(hours=12)
    with store.pool.connection() as conn:
        conn.execute(
            """UPDATE subscription_sheets
               SET next_due = %s, last_success = %s, last_failure = %s""",
            (due, before, before),
        )
    state = sheet(store, "B")
    with store.pool.connection() as conn:
        created, submitted = conn.execute(
            """SELECT created_at, last_submitted_at FROM sheet_subscriptions
               WHERE google_sheet_id = 'B'"""
        ).fetchone()
    store.reconcile(INSTANCE, {"B", "C"})
    store.reconcile(INSTANCE, {"B", "C"})
    assert relationships(store) == {(OTHER, "A"), (INSTANCE, "B"), (INSTANCE, "C")}
    assert sheet(store, "A")[:2] == (due, False)
    assert sheet(store, "B") == state
    with store.pool.connection() as conn:
        new_created, new_submitted = conn.execute(
            """SELECT created_at, last_submitted_at FROM sheet_subscriptions
               WHERE google_sheet_id = 'B'"""
        ).fetchone()
    assert new_created == created
    assert new_submitted > submitted


def test_empty_sets_unknown_removals_and_resubscribe(store):
    store.reconcile(INSTANCE, set())
    store.unsubscribe(INSTANCE, "UNKNOWN")
    assert sheet(store, "UNKNOWN") is None
    store.reconcile(INSTANCE, {"A"})
    store.reconcile(OTHER, {"A"})
    due = sheet(store, "A")[0]
    store.reconcile(INSTANCE, set())
    assert sheet(store, "A")[:2] == (due, False)
    store.unsubscribe(OTHER, "A")
    assert relationships(store) == set()
    assert sheet(store, "A")[:2] == (None, True)
    state = sheet(store, "A")
    store.unsubscribe(OTHER, "A")
    store.reconcile(OTHER, set())
    assert sheet(store, "A") == state
    before = datetime.now(UTC)
    store.reconcile(INSTANCE, {"A"})
    assert sheet(store, "A")[1] is False
    assert before <= sheet(store, "A")[0] <= datetime.now(UTC)


def test_migrations_and_constraints(db_settings):
    migrate(db_settings)
    migrate(db_settings)
    with psycopg.connect(db_settings.conninfo(), autocommit=True) as conn:
        assert conn.execute(
            "SELECT version FROM subscription_schema_versions"
        ).fetchall() == [("001_initial.sql",)]
        for invalid in ["", "A B", "é", "a/b", "A" * 129]:
            with pytest.raises(psycopg.Error):
                conn.execute(
                    "INSERT INTO subscription_sheets(google_sheet_id) VALUES (%s)",
                    (invalid,),
                )
        conn.execute("INSERT INTO subscription_sheets(google_sheet_id) VALUES ('A')")
        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute(
                "INSERT INTO subscription_sheets(google_sheet_id) VALUES ('A')"
            )
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            conn.execute(
                """INSERT INTO sheet_subscriptions(instance_key, google_sheet_id)
                   VALUES (%s, 'A')""",
                (b"x" * 32,),
            )
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                """UPDATE subscription_sheets
                   SET pending_delete = true, next_due = now() WHERE google_sheet_id = 'A'"""
            )
        conn.execute(
            """INSERT INTO subscription_runs(google_sheet_id, operation, status)
               VALUES ('A', 'refresh', 'running')"""
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute(
                """INSERT INTO subscription_runs(google_sheet_id, operation, status)
                   VALUES ('A', 'delete', 'running')"""
            )
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute("UPDATE subscription_runs SET status = 'succeeded'")
        conn.execute(
            "UPDATE subscription_runs SET status = 'succeeded', finished_at = clock_timestamp()"
        )
        conn.execute(
            """INSERT INTO subscription_runs(google_sheet_id, operation, status)
               VALUES ('A', 'delete', 'running')"""
        )
        run_count = conn.execute("SELECT count(*) FROM subscription_runs").fetchone()
        assert run_count is not None and run_count[0] == 2


def test_relationship_unique_constraint(store):
    store.reconcile(INSTANCE, {"A"})
    with store.pool.connection() as conn, pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(
            """INSERT INTO sheet_subscriptions
               SELECT instance_key, google_sheet_id, created_at, last_submitted_at
               FROM sheet_subscriptions"""
        )


def test_reconcile_rolls_back_every_change(store):
    store.reconcile(INSTANCE, {"A"})
    original = sheet(store, "A")
    with store.pool.connection() as conn:
        conn.execute(
            """CREATE FUNCTION reject_test_subscription() RETURNS trigger LANGUAGE plpgsql AS $$
               BEGIN RAISE EXCEPTION 'injected test failure'; END $$;
               CREATE TRIGGER reject_test_subscription BEFORE INSERT ON sheet_subscriptions
               FOR EACH ROW WHEN (NEW.google_sheet_id = 'Z_FAIL')
               EXECUTE FUNCTION reject_test_subscription()"""
        )
    with pytest.raises(psycopg.errors.RaiseException):
        store.reconcile(INSTANCE, {"B", "Z_FAIL"})
    assert relationships(store) == {(INSTANCE, "A")}
    assert sheet(store, "A") == original
    assert sheet(store, "B") is None
    assert sheet(store, "Z_FAIL") is None


def test_migration_transaction_rolls_back(db_settings, tmp_path, monkeypatch):
    import subscriptions.migrate as module

    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "001_broken.sql").write_text(
        "CREATE TABLE should_roll_back (id int); SELECT missing_column;",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "files", lambda package: tmp_path)
    with pytest.raises(psycopg.errors.UndefinedColumn):
        migrate(db_settings)
    with psycopg.connect(db_settings.conninfo()) as conn:
        assert conn.execute(
            "SELECT to_regclass('should_roll_back'), to_regclass('subscription_schema_versions')"
        ).fetchone() == (None, None)


def test_state_survives_resource_reinitialization(db_settings):
    migrate(db_settings)
    with open_subscription_store(db_settings) as first:
        first.reconcile(INSTANCE, {"A", "B"})
        first.unsubscribe(INSTANCE, "A")
        previous = sheet(first, "B")
        with first.pool.connection() as conn:
            conn.execute(
                """INSERT INTO subscription_runs
                   (google_sheet_id, operation, status, finished_at)
                   VALUES ('B', 'refresh', 'succeeded', clock_timestamp())"""
            )
    with open_subscription_store(db_settings) as second:
        assert relationships(second) == {(INSTANCE, "B")}
        assert sheet(second, "A")[:2] == (None, True)
        assert sheet(second, "B") == previous
        with second.pool.connection() as conn:
            run_count = conn.execute(
                "SELECT count(*) FROM subscription_runs"
            ).fetchone()
            assert run_count is not None and run_count[0] == 1


def test_unmigrated_database_fails_startup(db_settings):
    with (
        pytest.raises(psycopg.errors.UndefinedTable),
        open_subscription_store(db_settings),
    ):
        pytest.fail("unmigrated database must not open")


def test_same_instance_replacements_wait_for_database_lock(store, db_settings):
    store.reconcile(INSTANCE, {"A"})
    with (
        open_subscription_store(db_settings) as other,
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        with store.pool.connection() as conn, conn.transaction():
            SubscriptionStore._lock_instance(conn, INSTANCE)
            first = executor.submit(store.reconcile, INSTANCE, {"B", "C"})
            second = executor.submit(other.reconcile, INSTANCE, {"D"})
            deadline = monotonic() + 5
            while True:
                waiting = conn.execute(
                    """SELECT count(*) FROM pg_stat_activity
                       WHERE datname = current_database()
                       AND application_name = 'hia-subscriptions'
                       AND wait_event_type = 'Lock'"""
                ).fetchone()[0]
                if waiting == 2:
                    break
                assert monotonic() < deadline, (
                    "both workers must wait on PostgreSQL locks"
                )
                conn.execute("SELECT pg_stat_clear_snapshot()")
                sleep(0.01)
        first.result(timeout=10)
        second.result(timeout=10)
    assert relationships(store) in [
        {(INSTANCE, "B"), (INSTANCE, "C")},
        {(INSTANCE, "D")},
    ]
    assert sheet(store, "A")[:2] == (None, True)


def test_concurrent_shared_sheet_removal_and_addition(store, db_settings):
    # Independent pools represent different API worker processes.
    with (
        open_subscription_store(db_settings) as other,
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        for _ in range(10):
            store.reconcile(INSTANCE, {"A", "B"})
            other.reconcile(OTHER, set())
            gate = Barrier(2)

            def remove(gate=gate):
                gate.wait(timeout=5)
                store.reconcile(INSTANCE, set())

            def add(gate=gate):
                gate.wait(timeout=5)
                other.reconcile(OTHER, {"B", "A"})

            first, second = executor.submit(remove), executor.submit(add)
            first.result(timeout=10)
            second.result(timeout=10)
            assert relationships(store) == {(OTHER, "A"), (OTHER, "B")}
            assert all(sheet(store, sheet_id)[0] is not None for sheet_id in ("A", "B"))
            assert all(sheet(store, sheet_id)[1] is False for sheet_id in ("A", "B"))


def test_public_api_normalization_and_duplicates(subscription_client, store):
    client = subscription_client
    response = client.post(
        "/v1/subscriptions",
        json={
            "instance": "HTTPS://EXAMPLE.ORG:443/client?q=1#fragment",
            "googleSheetIds": ["B", "A", "B"],
        },
    )
    assert response.status_code == 200
    assert response.json() == {"instance": INSTANCE, "googleSheetIds": ["A", "B"]}
    due = sheet(store, "B")
    assert (
        client.post(
            "/v1/subscriptions",
            json={
                "instance": INSTANCE,
                "googleSheetIds": ["B"],
            },
        ).status_code
        == 200
    )
    assert relationships(store) == {(INSTANCE, "B")}
    assert sheet(store, "B") == due
    assert sheet(store, "A")[:2] == (None, True)
    assert (
        client.post(
            "/v1/subscriptions",
            json={
                "instance": INSTANCE,
                "googleSheetIds": [],
            },
        ).status_code
        == 200
    )
    assert relationships(store) == set()


@pytest.mark.parametrize("key", [None, "", "wrong", "test-api-key"])
def test_unsubscribe_requires_write_key(subscription_client, store, key):
    store.reconcile(INSTANCE, {"A"})
    headers = {"Authorization": key} if key is not None else {}
    response = subscription_client.post(
        "/v1/subscriptions/unsubscribe",
        headers=headers,
        json={
            "instance": INSTANCE,
            "googleSheetId": "A",
        },
    )
    assert response.status_code == 401
    assert relationships(store) == {(INSTANCE, "A")}
    assert sheet(store, "A")[1] is False


def test_authorized_unsubscribe_only_removes_pair(subscription_client, store):
    store.reconcile(INSTANCE, {"A", "B"})
    store.reconcile(OTHER, {"A"})
    client = subscription_client
    for sheet_id in ("UNKNOWN", "A", "A"):
        response = client.post(
            "/v1/subscriptions/unsubscribe",
            headers={"Authorization": "test-write-key"},
            json={
                "instance": "HTTPS://EXAMPLE.ORG:443/client?lang=en#section",
                "googleSheetId": sheet_id,
            },
        )
        assert response.status_code == 200
    assert relationships(store) == {(INSTANCE, "B"), (OTHER, "A")}
    assert sheet(store, "A")[1] is False
    assert sheet(store, "UNKNOWN") is None
    assert (
        client.post(
            "/v1/subscriptions/unsubscribe",
            headers={"Authorization": "test-write-key"},
            json={"instance": OTHER, "googleSheetId": "A"},
        ).status_code
        == 200
    )
    assert sheet(store, "A")[:2] == (None, True)


@pytest.mark.parametrize(
    "changes",
    [
        {"googleSheetIds": ["VALID", "bad/id"]},
        {"googleSheetIds": ["VALID", 42]},
        {"googleSheetIds": None},
        {"googleSheetIds": "A"},
        {"instance": "http://example.org", "googleSheetIds": []},
        {"instance": "https://example.org/" + "a" * 2083, "googleSheetIds": []},
        {"missing": True},
    ],
)
def test_invalid_requests_never_partially_reconcile(
    subscription_client, store, changes
):
    store.reconcile(INSTANCE, {"A", "B"})
    state = [sheet(store, "A"), sheet(store, "B")]
    payload = {"instance": INSTANCE, "googleSheetIds": ["VALID"]}
    payload.update(changes)
    if changes.get("missing"):
        del payload["googleSheetIds"]
    assert (
        subscription_client.post("/v1/subscriptions", json=payload).status_code == 422
    )
    assert relationships(store) == {(INSTANCE, "A"), (INSTANCE, "B")}
    assert [sheet(store, "A"), sheet(store, "B")] == state
    assert sheet(store, "VALID") is None


def test_paths_remain_distinct_and_unicode_urls_fit_storage(subscription_client, store):
    for instance in (
        "https://example.org/A",
        "https://example.org/a",
        "https://example.org/a/",
        "https://example.org/" + "界" * 100,
    ):
        assert (
            subscription_client.post(
                "/v1/subscriptions",
                json={
                    "instance": instance,
                    "googleSheetIds": ["A"],
                },
            ).status_code
            == 200
        )
    assert len(relationships(store)) == 4
