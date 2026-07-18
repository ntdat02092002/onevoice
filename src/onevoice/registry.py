from __future__ import annotations

from collections.abc import Callable
from typing import Any


class BackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, dict[str, Callable[..., Any]]] = {}

    def register(self, kind: str, name: str, factory: Callable[..., Any]) -> None:
        bucket = self._factories.setdefault(kind, {})
        if name in bucket:
            raise ValueError(f"Backend already registered: {kind}/{name}")
        bucket[name] = factory

    def create(self, kind: str, name: str, **kwargs: Any) -> Any:
        try:
            factory = self._factories[kind][name]
        except KeyError as exc:
            available = sorted(self._factories.get(kind, {}))
            raise ValueError(f"Unknown {kind} backend {name!r}; available: {available}") from exc
        return factory(**kwargs)

    def names(self, kind: str) -> tuple[str, ...]:
        return tuple(sorted(self._factories.get(kind, {})))


registry = BackendRegistry()
