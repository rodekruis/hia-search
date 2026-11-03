from collections.abc import Generator
from contextlib import contextmanager
from hashlib import sha256
from importlib.resources import files

from psycopg import Connection
from psycopg_pool import ConnectionPool

from subscriptions.settings import SubscriptionSettings


class SubscriptionStore:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    @staticmethod
    def _lock_instance(conn: Connection, instance: str) -> bytes:
        key = sha256(instance.encode("utf-8")).digest()
        conn.execute(
            """INSERT INTO subscription_instances(instance_key, instance)
               VALUES (%s, %s) ON CONFLICT DO NOTHING""",
            (key, instance),
        )
        row = conn.execute(
            "SELECT instance FROM subscription_instances WHERE instance_key = %s FOR UPDATE",
            (key,),
        ).fetchone()
        if row is None or row[0] != instance:
            raise RuntimeError("Subscription instance identity collision")
        return key

    @staticmethod
    def _lock_sheets(conn: Connection, sheet_ids: set[str]) -> None:
        # All writers (including the future dispatcher) must lock sheets in this order.
        for sheet_id in sorted(sheet_ids):
            conn.execute(
                """INSERT INTO subscription_sheets(google_sheet_id)
                   VALUES (%s) ON CONFLICT DO NOTHING""",
                (sheet_id,),
            )
            conn.execute(
                """SELECT google_sheet_id FROM subscription_sheets
                   WHERE google_sheet_id = %s FOR UPDATE""",
                (sheet_id,),
            )

    @staticmethod
    def _remove(conn: Connection, key: bytes, sheet_id: str) -> None:
        removed = conn.execute(
            """DELETE FROM sheet_subscriptions
               WHERE instance_key = %s AND google_sheet_id = %s
               RETURNING google_sheet_id""",
            (key, sheet_id),
        ).fetchone()
        if removed is not None:
            conn.execute(
                """UPDATE subscription_sheets SET next_due = NULL, pending_delete = true
                   WHERE google_sheet_id = %s AND NOT EXISTS (
                       SELECT 1 FROM sheet_subscriptions WHERE google_sheet_id = %s
                   )""",
                (sheet_id, sheet_id),
            )

    def reconcile(self, instance: str, sheet_ids: set[str]) -> None:
        with self.pool.connection() as conn, conn.transaction():
            key = self._lock_instance(conn, instance)
            existing = {
                row[0]
                for row in conn.execute(
                    "SELECT google_sheet_id FROM sheet_subscriptions WHERE instance_key = %s",
                    (key,),
                )
            }
            self._lock_sheets(conn, existing | sheet_ids)
            for sheet_id in sorted(existing - sheet_ids):
                self._remove(conn, key, sheet_id)
            for sheet_id in sorted(sheet_ids):
                conn.execute(
                    """INSERT INTO sheet_subscriptions(instance_key, google_sheet_id)
                       VALUES (%s, %s) ON CONFLICT (instance_key, google_sheet_id)
                       DO UPDATE SET last_submitted_at = clock_timestamp()""",
                    (key, sheet_id),
                )
                conn.execute(
                    """UPDATE subscription_sheets
                       SET next_due = COALESCE(next_due, clock_timestamp()),
                           pending_delete = false
                       WHERE google_sheet_id = %s""",
                    (sheet_id,),
                )

    def unsubscribe(self, instance: str, sheet_id: str) -> None:
        with self.pool.connection() as conn, conn.transaction():
            key = self._lock_instance(conn, instance)
            existing = conn.execute(
                """SELECT google_sheet_id FROM sheet_subscriptions
                   WHERE instance_key = %s AND google_sheet_id = %s""",
                (key, sheet_id),
            ).fetchone()
            if existing is not None:
                self._lock_sheets(conn, {sheet_id})
                self._remove(conn, key, sheet_id)


@contextmanager
def open_subscription_store(
    settings: SubscriptionSettings | None = None,
) -> Generator[SubscriptionStore, None, None]:
    settings = settings if settings is not None else SubscriptionSettings()
    with ConnectionPool(
        conninfo=settings.conninfo(),
        min_size=settings.pool_min_size,
        max_size=settings.pool_max_size,
        timeout=settings.timeout,
        open=False,
    ) as pool:
        pool.wait(timeout=settings.timeout)
        # Fail boot on missing/outdated schema; migrations never run in API workers.
        with pool.connection() as conn:
            versions = conn.execute(
                "SELECT version FROM subscription_schema_versions ORDER BY version"
            ).fetchall()
            migrations = sorted(
                path.name
                for path in files("subscriptions").joinpath("migrations").iterdir()
                if path.name.endswith(".sql")
            )
            applied = [version[0] for version in versions]
            if not migrations or applied != migrations:
                raise RuntimeError("Run subscription migrations before API startup")
        yield SubscriptionStore(pool)
