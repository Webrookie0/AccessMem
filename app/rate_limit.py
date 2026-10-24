"""Bounded per-process sliding-window limits for local deployment."""

import time
from collections import OrderedDict, deque
from math import ceil
from threading import Lock

from fastapi import Depends, HTTPException, Request

from .auth.jwt import get_current_user
from .auth.models import User
from .config import get_settings


class RateLimiter:
    def __init__(self):
        self._buckets: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = Lock()

    def check(self, key: str, limit: int) -> None:
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            while bucket and bucket[0] <= now - 60:
                bucket.popleft()
            self._buckets.move_to_end(key)
            if len(bucket) >= limit:
                raise HTTPException(
                    429,
                    "Too many requests. Please try again later.",
                    headers={"Retry-After": str(max(1, ceil(60 - (now - bucket[0]))))},
                )
            bucket.append(now)
            while len(self._buckets) > 10000:
                self._buckets.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()


limiter = RateLimiter()


def limit_login(request: Request) -> None:
    host = request.client.host if request.client else "unknown"
    limiter.check(f"login:{host}", get_settings().login_requests_per_minute)


def limit_chat(user: User = Depends(get_current_user)) -> None:
    limiter.check(f"chat:{user.tenant_id}:{user.username}", get_settings().chat_requests_per_minute)
