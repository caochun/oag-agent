"""Runtime contract for modeled, state-changing business Actions."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ActionRuntime(Protocol):
    """Execute the Action lifecycle without exposing its private effect DSL."""

    def list_actions(self, context_id: str = "") -> dict[str, Any]: ...

    def prepare_action(
        self,
        action_id: str,
        context_id: str = "",
        initial_inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def preview_action(
        self,
        action_id: str,
        inputs: dict[str, Any] | None = None,
        context_id: str = "",
    ) -> dict[str, Any]: ...

    def execute_action(
        self,
        preview_token: str,
        reason: str = "",
        actor: str = "",
        channel: str = "",
    ) -> dict[str, Any]: ...
