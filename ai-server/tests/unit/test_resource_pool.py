"""Unit tests for Resource Pool"""

import asyncio
import pytest

from services.resource_pool import ResourcePool


class TestResourcePool:
    """Test resource pool behavior"""

    @pytest.mark.asyncio
    async def test_pool_allows_concurrent_requests(self):
        """Test that pool allows requests up to max_concurrent"""
        pool = ResourcePool(max_concurrent=2, timeout=1.0)

        async def dummy_task(delay=0.1):
            async with pool:
                await asyncio.sleep(delay)
                return "done"

        # Run 2 concurrent tasks (should succeed)
        results = await asyncio.gather(
            dummy_task(),
            dummy_task()
        )

        assert results == ["done", "done"]
        assert pool.get_stats()["total_requests"] == 2

    @pytest.mark.asyncio
    async def test_pool_blocks_when_full(self):
        """Test that pool blocks when at capacity"""
        pool = ResourcePool(max_concurrent=1, timeout=5.0)

        task1_started = asyncio.Event()
        task2_waited = False

        async def long_task():
            async with pool:
                task1_started.set()
                await asyncio.sleep(0.2)

        async def blocking_task():
            nonlocal task2_waited
            await task1_started.wait()
            # Pool should be full now
            if pool._semaphore.locked():
                task2_waited = True
            async with pool:
                return "done"

        # Run tasks concurrently
        results = await asyncio.gather(
            long_task(),
            blocking_task()
        )

        assert task2_waited
        assert pool.get_stats()["pool_waits"] == 1

    @pytest.mark.asyncio
    async def test_pool_timeout(self):
        """Test that pool times out when waiting too long"""
        pool = ResourcePool(max_concurrent=1, timeout=0.1)

        async def blocking_task():
            async with pool:
                await asyncio.sleep(1.0)  # Hold slot for a while

        async def timeout_task():
            await asyncio.sleep(0.05)  # Wait for first task to acquire
            async with pool:  # Should timeout
                return "done"

        # First task acquires slot
        task1 = asyncio.create_task(blocking_task())

        # Second task should timeout
        with pytest.raises(RuntimeError, match="Resource pool timeout"):
            await timeout_task()

        # Cancel first task to clean up
        task1.cancel()
        try:
            await task1
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_pool_releases_on_exception(self):
        """Test that pool slot is released even when exception occurs"""
        pool = ResourcePool(max_concurrent=1, timeout=1.0)

        async def failing_task():
            async with pool:
                raise ValueError("Test error")

        # First task fails
        with pytest.raises(ValueError):
            await failing_task()

        # Pool should be released, second task should work
        async def success_task():
            async with pool:
                return "success"

        result = await success_task()
        assert result == "success"
        assert pool.get_stats()["active_requests"] == 0

    @pytest.mark.asyncio
    async def test_pool_stats(self):
        """Test that pool statistics are tracked correctly"""
        pool = ResourcePool(max_concurrent=3, timeout=1.0)

        async def task():
            async with pool:
                await asyncio.sleep(0.01)

        # Run some tasks
        await asyncio.gather(task(), task(), task())

        stats = pool.get_stats()
        assert stats["max_concurrent"] == 3
        assert stats["total_requests"] == 3
        assert stats["active_requests"] == 0
        assert stats["available_slots"] == 3

    @pytest.mark.asyncio
    async def test_pool_concurrent_limit(self):
        """Test that pool never exceeds max_concurrent"""
        pool = ResourcePool(max_concurrent=2, timeout=5.0)
        max_concurrent_seen = 0

        async def monitored_task():
            nonlocal max_concurrent_seen
            async with pool:
                current = pool._active_requests
                max_concurrent_seen = max(max_concurrent_seen, current)
                await asyncio.sleep(0.1)

        # Run 5 tasks that will queue
        await asyncio.gather(*[monitored_task() for _ in range(5)])

        # Should never exceed max_concurrent
        assert max_concurrent_seen <= 2
        assert pool.get_stats()["total_requests"] == 5
