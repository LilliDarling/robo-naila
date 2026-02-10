from __future__ import annotations

import logging

import aiohttp

log = logging.getLogger(__name__)


async def exchange_sdp(hub_url: str, device_id: str, offer_sdp: str) -> str:
    """POST the SDP offer to the hub and return the SDP answer.

    Matches hub/src/http.rs ConnectRequest / ConnectResponse:
        Request:  {"device_id": "...", "sdp": "..."}
        Response: {"sdp": "..."}
    """
    url = f"{hub_url}/connect"
    payload = {"device_id": device_id, "sdp": offer_sdp}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"SDP exchange failed ({resp.status}): {body}")
            data = await resp.json()
            answer_sdp = data["sdp"]
            log.info("SDP exchange OK (%d chars answer)", len(answer_sdp))
            return answer_sdp
