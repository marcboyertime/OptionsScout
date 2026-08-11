from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

try:
    import exchange_calendars as xcals  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - dependency health failure path
    xcals = None

NY = ZoneInfo("America/New_York")


def provider_status() -> dict[str, str]:
    if xcals is None:
        return {"provider": "exchange_calendars", "status": "UNAVAILABLE"}
    return {"provider": "exchange_calendars", "status": "AVAILABLE", "calendar": "XNYS"}


def session_status(at: datetime, product_calendar: str = "XNYS") -> dict[str, str | bool]:
    if xcals is None:
        return {"available": False, "reason": "verified exchange calendar provider unavailable"}
    try:
        calendar = xcals.get_calendar(product_calendar)
    except Exception:
        return {"available": False, "reason": f"unknown product calendar {product_calendar}"}
    local = at.astimezone(NY)
    session = local.date().isoformat()
    if not calendar.is_session(session):
        return {
            "available": True,
            "regular": False,
            "session": "CLOSED",
            "reason": "weekend or exchange holiday",
        }
    open_time = calendar.session_open(session).to_pydatetime().astimezone(NY)
    close_time = calendar.session_close(session).to_pydatetime().astimezone(NY)
    regular = open_time <= local <= close_time
    period = "REGULAR" if regular else "PRE_OR_AFTER"
    return {
        "available": True,
        "regular": regular,
        "session": period,
        "open": open_time.isoformat(),
        "close": close_time.isoformat(),
    }
