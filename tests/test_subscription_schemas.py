import pytest
from pydantic import ValidationError

from routes.subscription_schemas import SubscriptionPayload, UnsubscribePayload
from subscriptions.settings import SubscriptionSettings


@pytest.mark.parametrize(
    "instance",
    [
        None,
        123,
        {},
        "",
        "http://example.org",
        "/relative",
        "example.org",
        "ftp://example.org",
        "https:example.org",
        "https:///example.org",
        "https://",
        "https://[invalid",
        "https://user@example.org",
        "https://user:pass@example.org",
        "https://@example.org",
        " https://example.org",
        "https://example.org ",
        "https://example.org/a b",
        "https://example.org/?q=a\nb",
        "https://example.org/#a\tb",
    ],
)
def test_invalid_instance(instance):
    with pytest.raises(ValidationError):
        SubscriptionPayload(instance=instance, googleSheetIds=["A"])


@pytest.mark.parametrize(
    ("instance", "normalized"),
    [
        ("HTTPS://EXAMPLE.ORG:443", "https://example.org/"),
        ("https://example.org:8443/path", "https://example.org:8443/path"),
        (
            "https://example.org/client/site?lang=en#section",
            "https://example.org/client/site",
        ),
        ("https://example.org/A/b/c/d/e/", "https://example.org/A/b/c/d/e/"),
        ("https://example.org/A/b/c/d/e", "https://example.org/A/b/c/d/e"),
        ("https://example.org/?x=/a#fragment", "https://example.org/"),
    ],
)
def test_normalized_instance(instance, normalized):
    assert (
        SubscriptionPayload(instance=instance, googleSheetIds=[]).normalized_instance
        == normalized
    )
    assert (
        UnsubscribePayload(instance=instance, googleSheetId="A").normalized_instance
        == normalized
    )


@pytest.mark.parametrize("suffix", ["/", "/?q=", "/#"])
def test_httpurl_builtin_length_limit(suffix):
    prefix = "https://example.org" + suffix
    at_limit = prefix + "a" * (2083 - len(prefix))
    SubscriptionPayload(instance=at_limit, googleSheetIds=[])
    with pytest.raises(ValidationError):
        SubscriptionPayload(instance=at_limit + "a", googleSheetIds=[])


@pytest.mark.parametrize(
    "sheet_id",
    [
        "",
        "A" * 129,
        "https://docs.google.com/spreadsheets/d/A",
        "A B",
        " A",
        "A\n",
        "a/b",
        "a?b",
        "a#b",
        "é",
        "a.b",
        1,
        None,
        {},
    ],
)
def test_invalid_sheet_id(sheet_id):
    with pytest.raises(ValidationError):
        SubscriptionPayload(
            instance="https://example.org", googleSheetIds=["VALID", sheet_id]
        )
    with pytest.raises(ValidationError):
        UnsubscribePayload(instance="https://example.org", googleSheetId=sheet_id)


@pytest.mark.parametrize("sheet_id", ["a", "Z" * 128, "Aa09_-"])
def test_valid_sheet_ids(sheet_id):
    assert SubscriptionPayload(
        instance="https://example.org", googleSheetIds=[sheet_id]
    ).googleSheetIds == [sheet_id]


@pytest.mark.parametrize("value", [None, "A", {}, 5, [None], [1]])
def test_list_is_explicit_and_strict(value):
    with pytest.raises(ValidationError):
        SubscriptionPayload(instance="https://example.org", googleSheetIds=value)


def test_missing_list_rejected():
    with pytest.raises(ValidationError):
        SubscriptionPayload.model_validate({"instance": "https://example.org"})


def test_subscription_settings_are_independent(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DB_HOST", "checkpoint.invalid")
    monkeypatch.setenv("CHECKPOINT_DB_PASSWORD", "checkpoint-only")
    settings = SubscriptionSettings(
        host="subscriptions.invalid",
        name="subscription_test",
        user="subscription_test",
        password="subscription-only",
    )
    assert "subscriptions.invalid" in settings.conninfo()
    assert "checkpoint" not in settings.conninfo()
    assert "subscription-only" not in repr(settings)
    with pytest.raises(ValidationError):
        SubscriptionSettings(pool_min_size=5, pool_max_size=1)
