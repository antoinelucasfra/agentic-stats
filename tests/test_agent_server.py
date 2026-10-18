"""The agent panel's guards: a turn budget and a concurrency cap.

A public instance cannot let one conversation run forever or three slow streams
starve it. Both guards are plain Python around the turn, so they are tested
without a model and without a browser.
"""

from __future__ import annotations

import asyncio

from web.panels.agent import server as agent_server


def test_slots_refuse_rather_than_queue():
    async def main() -> None:
        await agent_server._SLOTS.acquire()
        await agent_server._SLOTS.acquire()
        await agent_server._SLOTS.acquire()
        assert await agent_server._take_slot() is False
        agent_server._SLOTS.release()
        assert await agent_server._take_slot() is True
        agent_server._SLOTS.release()

    asyncio.run(main())


def test_the_budget_and_the_cap_are_named_constants():
    assert agent_server.TURNS_PER_SESSION == 20
    assert agent_server.MAX_CONCURRENT_TURNS == 3
