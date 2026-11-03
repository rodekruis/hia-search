"""Opt-in tests use only a loopback disposable server with fixed test credentials."""

import os
import socket
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql

from main import create_app
from subscriptions.migrate import migrate
from subscriptions.settings import SubscriptionSettings
from subscriptions.store import open_subscription_store


@pytest.fixture()
def db_settings():
    if os.environ.get("SUBSCRIPTION_TEST_POSTGRES") != "1":
        pytest.skip(
            "Set SUBSCRIPTION_TEST_POSTGRES=1 with the disposable loopback PostgreSQL server"
        )
    admin = SubscriptionSettings(
        host="127.0.0.1",
        port=int(os.environ.get("SUBSCRIPTION_TEST_PORT", "55432")),
        name="subscription_test",
        user="subscription_test",
        password="disposable-test-only",
        sslmode="disable",
        pool_min_size=1,
        pool_max_size=5,
        timeout=5,
    )
    name = "subscription_test_" + uuid4().hex
    # Each test owns a new database; never use application or checkpoint settings.
    with psycopg.connect(admin.conninfo(), autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    settings = admin.model_copy(update={"name": name})
    try:
        yield settings
    finally:
        with psycopg.connect(admin.conninfo(), autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
            )


@pytest.fixture()
def store(db_settings):
    migrate(db_settings)
    with open_subscription_store(db_settings) as resource:
        yield resource


@pytest.fixture()
def subscription_client(store, monkeypatch):
    # Python HTTP clients must not reach Sheets, Azure, embeddings or translation.
    # Psycopg's libpq connection is independent of Python's socket implementation.
    def reject_network(*args, **kwargs):
        raise AssertionError("Subscription requests must not access external services")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    from contextlib import nullcontext

    app = create_app(subscription_store_factory=lambda: nullcontext(store))
    with TestClient(app) as client:
        yield client
