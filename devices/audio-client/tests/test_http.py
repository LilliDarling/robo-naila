import pytest
import aiohttp
from aiohttp import web

from audio_client.http import exchange_sdp


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
    async with aiohttp.ClientSession() as session:
        answer = await exchange_sdp(session, url, "pi-test", "offer-sdp-data")
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
    async with aiohttp.ClientSession() as session:
        with pytest.raises(RuntimeError, match="500"):
            await exchange_sdp(session, url, "pi-test", "offer")


@pytest.mark.asyncio
async def test_trace_header_is_optional_and_forwarded(aiohttp_server):
    received = []

    async def connect(request):
        received.append(request.headers.get('traceparent'))
        return web.json_response({'sdp': 'answer'})

    app = web.Application()
    app.router.add_post('/connect', connect)
    server = await aiohttp_server(app)
    async with aiohttp.ClientSession() as session:
        await exchange_sdp(session, str(server.make_url('')).rstrip('/'), 'a', 'offer')
        await exchange_sdp(session, str(server.make_url('')).rstrip('/'), 'b', 'offer', trace_header='00-test')
    assert received == [None, '00-test']
