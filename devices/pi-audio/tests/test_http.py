import pytest
from aiohttp import web

from pi_audio.http import exchange_sdp


@pytest.fixture
def hub_server(aiohttp_server):
    """Spin up a mock hub that echoes the SDP back with a prefix."""

    async def connect_handler(request: web.Request) -> web.Response:
        body = await request.json()
        assert "device_id" in body
        assert "sdp" in body
        return web.json_response({"sdp": "answer:" + body["sdp"]})

    app = web.Application()
    app.router.add_post("/connect", connect_handler)
    return aiohttp_server(app)


@pytest.mark.asyncio
async def test_exchange_sdp_success(hub_server):
    server = await hub_server
    url = f"http://{server.host}:{server.port}"
    answer = await exchange_sdp(url, "pi-test", "offer-sdp-data")
    assert answer == "answer:offer-sdp-data"


@pytest.fixture
def hub_server_error(aiohttp_server):
    async def connect_handler(request: web.Request) -> web.Response:
        return web.Response(status=500, text="internal error")

    app = web.Application()
    app.router.add_post("/connect", connect_handler)
    return aiohttp_server(app)


@pytest.mark.asyncio
async def test_exchange_sdp_error(hub_server_error):
    server = await hub_server_error
    url = f"http://{server.host}:{server.port}"
    with pytest.raises(RuntimeError, match="500"):
        await exchange_sdp(url, "pi-test", "offer")
