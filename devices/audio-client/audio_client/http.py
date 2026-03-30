from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from .profile import DeviceProfile

log = logging.getLogger(__name__)


async def exchange_sdp(
    session: aiohttp.ClientSession,
    hub_url: str,
    device_id: str,
    offer_sdp: str,
    profile: DeviceProfile | None = None,
) -> str:
    """POST the SDP offer to the hub and return the SDP answer.

    Matches hub/src/http.rs ConnectRequest / ConnectResponse:
        Request:  {"device_id": "...", "sdp": "...", ...capabilities}
        Response: {"sdp": "..."}
    """
    url = f"{hub_url}/connect"
    payload: dict = {"device_id": device_id, "sdp": offer_sdp}

    if profile is not None:
        payload["preferred_output_sample_rate"] = profile.preferred_output_sample_rate
        payload["supported_output_codecs"] = [c.value for c in profile.supported_output_codecs]

    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"SDP exchange failed ({resp.status}): {body}")
        data = await resp.json()
        answer_sdp = data["sdp"]
        log.info("SDP exchange OK (%d chars answer)", len(answer_sdp))
        return answer_sdp
