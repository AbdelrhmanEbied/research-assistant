from fastapi import APIRouter, Depends, HTTPException, Query, Request

from telemetry.storage import TelemetryStore, get_default_store

router = APIRouter(
    prefix="/telemetry",
    tags=["Telemetry"],
)


def get_store(request: Request) -> TelemetryStore:
    store = getattr(request.app.state, "telemetry_store", None)
    if store is None:
        store = get_default_store()
        request.app.state.telemetry_store = store
    return store


@router.get("/summary")
async def telemetry_summary(
    store: TelemetryStore = Depends(get_store),  # noqa: B008
):
    return store.summary()


@router.get("/events")
async def list_events(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    store: TelemetryStore = Depends(get_store),  # noqa: B008
):
    return {
        "events": store.list_events(limit=limit, offset=offset),
        "limit": limit,
        "offset": offset,
    }


@router.get("/events/{event_id}")
async def get_event(
    event_id: int,
    store: TelemetryStore = Depends(get_store),  # noqa: B008
):
    event = store.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Telemetry event not found.")
    return event


@router.delete("/events")
async def clear_events(
    store: TelemetryStore = Depends(get_store),  # noqa: B008
):
    deleted = store.clear()
    return {"deleted": deleted}
