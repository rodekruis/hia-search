"""Run explicitly with `uv run python -m subscriptions.migrate` before API startup."""

import logging
from importlib.resources import files
from typing import LiteralString, cast

import psycopg

from subscriptions.settings import SubscriptionSettings

logger = logging.getLogger(__name__)


def migrate(settings: SubscriptionSettings) -> None:
    with psycopg.connect(settings.conninfo()) as conn, conn.transaction():
        # A separate namespace from API row locks; serializes migration runners.
        conn.execute("SELECT pg_advisory_xact_lock(510, 1)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS subscription_schema_versions (
                version text PRIMARY KEY,
                applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
            )"""
        )
        applied = {
            row[0]
            for row in conn.execute("SELECT version FROM subscription_schema_versions")
        }
        migrations = sorted(
            (
                path
                for path in files("subscriptions").joinpath("migrations").iterdir()
                if path.name.endswith(".sql")
            ),
            key=lambda path: path.name,
        )
        unknown = applied - {path.name for path in migrations}
        if unknown:
            raise RuntimeError("Subscription database has newer, unknown migrations")
        for path in migrations:
            if path.name not in applied:
                conn.execute(cast(LiteralString, path.read_text(encoding="utf-8")))
                conn.execute(
                    "INSERT INTO subscription_schema_versions(version) VALUES (%s)",
                    (path.name,),
                )
                logger.info(
                    "Applied subscription migration", extra={"version": path.name}
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate(SubscriptionSettings())
