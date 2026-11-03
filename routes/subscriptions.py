from typing import Annotated

from fastapi import APIRouter, Depends, Request

from routes.subscription_schemas import (
    SubscriptionPayload,
    SubscriptionResponse,
    UnsubscribePayload,
    UnsubscribeResponse,
)
from subscriptions.store import SubscriptionStore
from utils.auth import require_write_key

router = APIRouter(prefix="/v1/subscriptions", tags=["subscriptions"])


def get_subscription_store(request: Request) -> SubscriptionStore:
    return request.app.state.subscription_store


@router.post("", response_model=SubscriptionResponse, status_code=200)
def replace_subscriptions(
    payload: SubscriptionPayload,
    store: Annotated[SubscriptionStore, Depends(get_subscription_store)],
) -> SubscriptionResponse:
    """Publicly replace one instance's complete subscription set. No index I/O."""
    sheet_ids = set(payload.googleSheetIds)
    instance = payload.normalized_instance
    store.reconcile(instance, sheet_ids)
    return SubscriptionResponse(instance=instance, googleSheetIds=sorted(sheet_ids))


@router.post(
    "/unsubscribe",
    response_model=UnsubscribeResponse,
    status_code=200,
    dependencies=[Depends(require_write_key)],
)
def unsubscribe_sheet(
    payload: UnsubscribePayload,
    store: Annotated[SubscriptionStore, Depends(get_subscription_store)],
) -> UnsubscribeResponse:
    """Remove only this relationship, if present. Requires API_KEY_WRITE."""
    instance = payload.normalized_instance
    store.unsubscribe(instance, payload.googleSheetId)
    return UnsubscribeResponse(instance=instance, googleSheetId=payload.googleSheetId)
