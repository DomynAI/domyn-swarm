# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from typing import Any
import uuid


def new_session_id() -> str:
    """Return a random session ID as 32 lowercase hex characters."""
    return uuid.uuid4().hex


def session_request_options(
    session_id: str, extra_body: dict[str, Any], *, final: bool
) -> dict[str, Any]:
    """Build per-request options that pin a conversation to one session.

    Cache-aware routers keep every request that carries the same session ID on
    one replica. The ID goes in the ``X-Session-ID`` header and in the
    ``session_params.session_id`` body field. ``final`` also sets
    ``X-Dynamo-Session-Final: true`` so Dynamo can release the session.

    A ``session_params.session_id`` already present in ``extra_body`` wins over
    ``session_id`` in both the header and the body, so the two always carry the
    same ID. ``extra_body`` is copied, never mutated.

    Args:
        session_id: ID shared by all requests of one conversation, unless
            ``extra_body`` sets its own.
        extra_body: Request body fields from the job configuration.
        final: Whether this is the last request of the conversation.

    Returns:
        Keyword arguments ``extra_headers`` and ``extra_body`` for ``create()``.
    """
    session_params = {"session_id": session_id, **extra_body.get("session_params", {})}
    headers = {"X-Session-ID": session_params["session_id"]}
    if final:
        headers["X-Dynamo-Session-Final"] = "true"
    return {
        "extra_headers": headers,
        "extra_body": {**extra_body, "session_params": session_params},
    }
