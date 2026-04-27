"""Business logic for the /offload command.

Extracts the core offload workflow from the UI layer so it can be
tested independently of the Textual app.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AnyMessage, HumanMessage, get_buffer_string, message_to_dict
from langchain_core.messages.utils import (
    count_tokens_approximately,
    convert_to_messages,
)

from deepagents_cli.config import create_model
from deepagents_cli.textual_adapter import format_token_count

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.summarization import (
        SummarizationEvent,
        SummarizationMiddleware,
    )
    from langchain_core.messages import AnyMessage, HumanMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OffloadResult:
    """Successful offload result."""

    new_event: SummarizationEvent
    """The summarization event to write into agent state."""

    messages_offloaded: int
    """Number of older messages that were offloaded."""

    messages_kept: int
    """Number of recent messages retained in context."""

    tokens_before: int
    """Approximate token count of the conversation before offloading."""

    tokens_after: int
    """Approximate token count of the conversation after offloading."""

    pct_decrease: int
    """Percentage decrease in token usage."""

    offload_warning: str | None
    """Non-`None` when the backend write failed (non-fatal)."""


@dataclass(frozen=True)
class OffloadThresholdNotMet:
    """Offload was a no-op — conversation is within the retention budget."""

    conversation_tokens: int
    """Approximate token count of the conversation messages alone."""

    total_context_tokens: int
    """Total context token count including system overhead, or `0` when no
    token tracker is available."""

    context_limit: int | None
    """Model context window limit, if available."""

    budget_str: str
    """Human-readable retention budget (e.g. "20.0K tokens")."""


class OffloadModelError(Exception):
    """Raised when the model cannot be created for offloading."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_offload_limit(
    keep: tuple[str, int | float], context_limit: int | None
) -> str:
    """Format offload retention settings into a human-readable limit string.

    Args:
        keep: Retention policy tuple `(type, value)` from summarization
            defaults, where `type` is one of `"messages"`, `"tokens"`, or
            `"fraction"`.
        context_limit: Model context limit when available.

    Returns:
        A short display string describing the offload retention limit.
    """
    keep_type, keep_value = keep

    if keep_type == "messages":
        count = int(keep_value)
        noun = "message" if count == 1 else "messages"
        return f"last {count} {noun}"

    if keep_type == "tokens":
        return f"{format_token_count(int(keep_value))} tokens"

    if keep_type == "fraction":
        percent = float(keep_value) * 100
        if context_limit is not None:
            token_limit = max(1, int(context_limit * float(keep_value)))
            return f"{format_token_count(token_limit)} tokens"
        return f"{percent:.0f}% of context window"

    return "current retention threshold"


def _sanitize_message_content(messages: list[AnyMessage]) -> list[AnyMessage]:
    sanitized = []
    for msg in messages:
        if isinstance(msg.content, bytes):
            try:
                # Attempt to decode, fallback to placeholder
                decoded_content = msg.content.decode("utf-8")
            except UnicodeDecodeError:
                decoded_content = "<Binary content (offloaded to file)>"
                logger.warning("Binary content detected in message; replacing with placeholder.")
            sanitized.append(msg.__class__(
                content=decoded_content,
                additional_kwargs=msg.additional_kwargs,
                example=msg.example,
                name=msg.name,
                id=msg.id,
            ))
        else:
            sanitized.append(msg)
    return sanitized

def _normalize_messages(messages: list[Any]) -> list[Any]:
    """Convert OpenAI-style dict messages into LangChain message objects.

    The CLI can receive history from remote checkpoints or older state
    snapshots as plain dictionaries. DeepAgents' summarization middleware
    expects BaseMessage instances, so we normalize here before offloading.
    """
    if not messages:
        return messages

    if any(isinstance(message, dict) for message in messages):
        # LangChain's convert_to_messages expects 'role' in flat dicts, but
        # state snapshots might use 'type' (e.g. from pydantic serialization).
        normalized = []
        for msg in messages:
            if isinstance(msg, dict) and "type" in msg and "role" not in msg:
                # Map standard LC types to roles so convert_to_messages succeeds in environments
                # where the flat-dict-with-type format is not natively supported.
                role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
                m = msg.copy()
                m["role"] = role_map.get(msg["type"], msg["type"])
                normalized.append(m)
            else:
                normalized.append(msg)
        return convert_to_messages(normalized)

    return messages


def _coerce_summary_message(message: Any) -> HumanMessage:
    """Ensure summarization state stores a JSON-serializable `HumanMessage`.

    The summarization middleware may return dict-like payloads or message
    objects from different runtimes. We always normalize to a `HumanMessage`
    with primitive (JSON-safe) `additional_kwargs` because state checkpointing
    rejects nested message objects inside kwargs.
    """
    if isinstance(message, HumanMessage):
        content = message.content
        additional_kwargs = message.additional_kwargs
    elif isinstance(message, dict):
        content = message.get("content", "")
        additional_kwargs = message.get("additional_kwargs", {})
    else:
        content = getattr(message, "content", "")
        additional_kwargs = getattr(message, "additional_kwargs", None)

    if not isinstance(additional_kwargs, dict):
        additional_kwargs = {}

    sanitized_kwargs: dict[str, Any] = {}
    for key, value in additional_kwargs.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized_kwargs[key] = value
        elif isinstance(value, list):
            sanitized_items: list[Any] = []
            for item in value:
                if isinstance(item, AnyMessage):
                    sanitized_items.append(message_to_dict(item))
                else:
                    sanitized_items.append(item)
            sanitized_kwargs[key] = sanitized_items
        elif isinstance(value, tuple):
            sanitized_items = []
            for item in value:
                if isinstance(item, AnyMessage):
                    sanitized_items.append(message_to_dict(item))
                else:
                    sanitized_items.append(item)
            sanitized_kwargs[key] = sanitized_items
        elif isinstance(value, dict):
            sanitized_dict: dict[str, Any] = {}
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, AnyMessage):
                    sanitized_dict[nested_key] = message_to_dict(nested_value)
                else:
                    sanitized_dict[nested_key] = nested_value
            sanitized_kwargs[key] = sanitized_dict
        elif isinstance(value, AnyMessage):
            sanitized_kwargs[key] = message_to_dict(value)
        else:
            sanitized_kwargs[key] = str(value)

    return HumanMessage(content=cast(str, content), additional_kwargs=sanitized_kwargs)


async def offload_messages_to_backend(
    messages: list[Any],
    middleware: SummarizationMiddleware,
    *,
    thread_id: str,
    backend: BackendProtocol,
) -> str | None:
    """Write messages to backend storage before offloading.

    Appends messages as a timestamped markdown section to the conversation
    history file, matching the `SummarizationMiddleware` offload pattern.

    Filters out prior summary messages using the middleware's
    `_filter_summary_messages` to avoid storing summaries-of-summaries.

    Args:
        messages: Messages to offload.
        middleware: `SummarizationMiddleware` instance for filtering.
        thread_id: Thread identifier used to derive the storage path.
        backend: Backend to persist conversation history to.

    Returns:
        File path where history was stored, `""` (empty string) if there were no
            non-summary messages to offload (not an error), or `None` if the
            write failed.
    """
    path = f"/conversation_history/{thread_id}.md"
    fallback_path = os.path.expanduser(f"~/.deepagents/conversation_history/{thread_id}.md")

    # Exclude prior summaries so the offloaded history contains only
    # original messages
    normalized = _normalize_messages(messages)
    sanitized = _sanitize_message_content(normalized)
    filtered = middleware._filter_summary_messages(sanitized)
    if not filtered:
        return ""

    timestamp = datetime.now(UTC).isoformat()
    buf = get_buffer_string(filtered)
    new_section = f"## Offloaded at {timestamp}\n\n{buf}\n\n"

    existing_content = ""
    try:
        responses = await backend.adownload_files([path])
        resp = responses[0] if responses else None
        if resp and resp.content is not None and resp.error is None:
            existing_content = resp.content.decode("utf-8")
    except Exception as exc:  # abort write on read failure
        logger.warning(
            "Failed to read existing history at %s; aborting offload to "
            "avoid overwriting prior history: %s",
            path,
            exc,
            exc_info=True,
        )
        return None

    combined = existing_content + new_section

    try:
        result = (
            await backend.aedit(path, existing_content, combined)
            if existing_content
            else await backend.awrite(path, combined)
        )
        if result is None or result.error:
            error_detail = result.error if result else "backend returned None"
            logger.warning(
                "Failed to offload conversation history to %s: %s",
                path,
                error_detail,
            )
            fallback_result = await backend.awrite(fallback_path, combined)
            if fallback_result is None or fallback_result.error:
                logger.warning(
                    "Fallback offload also failed for %s: %s",
                    fallback_path,
                    fallback_result.error if fallback_result else "backend returned None",
                )
                return None
            logger.debug("Offloaded %d messages to fallback %s", len(filtered), fallback_path)
            return fallback_path
    except Exception as exc:  # defensive: surface write failures gracefully
        logger.warning(
            "Exception offloading conversation history to %s: %s",
            path,
            exc,
            exc_info=True,
        )
        try:
            fallback_result = await backend.awrite(fallback_path, combined)
            if fallback_result is None or fallback_result.error:
                logger.warning(
                    "Fallback offload also failed for %s: %s",
                    fallback_path,
                    fallback_result.error if fallback_result else "backend returned None",
                )
                return None
            logger.debug("Offloaded %d messages to fallback %s", len(filtered), fallback_path)
            return fallback_path
        except Exception as fallback_exc:
            logger.warning(
                "Exception offloading conversation history to fallback %s: %s",
                fallback_path,
                fallback_exc,
                exc_info=True,
            )
            return None

    logger.debug("Offloaded %d messages to %s", len(filtered), path)
    return path


# ---------------------------------------------------------------------------
# Core offload workflow
# ---------------------------------------------------------------------------


async def perform_offload(
    *,
    messages: list[Any],
    prior_event: SummarizationEvent | None,
    thread_id: str,
    model_spec: str,
    profile_overrides: dict[str, Any] | None,
    context_limit: int | None,
    total_context_tokens: int,
    backend: BackendProtocol | None,
) -> OffloadResult | OffloadThresholdNotMet:
    """Execute the offload workflow: summarize old messages and free context.

    Args:
        messages: Current conversation messages from agent state.

            May be LangChain message objects or serialized dicts (the latter
            when read from a remote HTTP state snapshot).
        prior_event: Existing `_summarization_event` if any.

            In server mode `summary_message` may be a serialized message dict.
        thread_id: Thread identifier for backend storage.
        model_spec: Model specification string (e.g. "openai:gpt-4").
        profile_overrides: Optional profile overrides from CLI flags.
        context_limit: Model context limit from settings.
        total_context_tokens: Current total context token count, or `0` when
            no token tracker is available.
        backend: Backend for persisting offloaded history.

    Returns:
        `OffloadResult` on success, `OffloadThresholdNotMet` when the
            conversation is within the retention budget.

    Raises:
        OffloadModelError: If the model cannot be created.
    """
    from deepagents.middleware.summarization import (
        SummarizationMiddleware,
        compute_summarization_defaults,
    )

    # Remote HTTP state snapshots may surface serialized message dicts instead
    # of LangChain message objects. Normalize them before passing state into
    # summarization middleware helpers. `any(...)` rather than checking index
    # 0 guards against heterogeneous lists (e.g. a snapshot with a streamed
    # append).
    needs_message_conversion = any(isinstance(m, dict) for m in messages)
    needs_summary_conversion = prior_event is not None and isinstance(
        prior_event.get("summary_message"), dict
    )
    if needs_message_conversion or needs_summary_conversion:
        from langchain_core.messages.utils import convert_to_messages

        if needs_message_conversion:
            messages = cast("list[AnyMessage]", convert_to_messages(messages))
        if needs_summary_conversion and prior_event is not None:
            converted_summary = cast(
                "HumanMessage",
                convert_to_messages([prior_event["summary_message"]])[0],
            )
            prior_event = {
                "cutoff_index": prior_event["cutoff_index"],
                "summary_message": converted_summary,
                "file_path": prior_event["file_path"],
            }

    try:
        result = create_model(model_spec, profile_overrides=profile_overrides)
        model = result.model
    except Exception as exc:
        msg = f"Offload requires a working model configuration: {exc}"
        raise OffloadModelError(msg) from exc

    # Patch context limit into model profile when it differs from the native
    # value (e.g. set via --profile-override or runtime config).
    if context_limit is not None:
        profile = getattr(model, "profile", None)
        native = profile.get("max_input_tokens") if isinstance(profile, dict) else None
        if native != context_limit:
            merged = (
                {**profile, "max_input_tokens": context_limit}
                if isinstance(profile, dict)
                else {"max_input_tokens": context_limit}
            )
            try:
                model.profile = merged  # type: ignore[union-attr]
            except (AttributeError, TypeError, ValueError):
                logger.warning(
                    "Could not patch context limit (%d) into model profile; "
                    "offload budget will use the model's native context window",
                    context_limit,
                    exc_info=True,
                )

    defaults = compute_summarization_defaults(model)
    offload_backend = backend
    if offload_backend is None:
        from deepagents.backends.filesystem import FilesystemBackend

        offload_backend = FilesystemBackend()
        logger.info("Using local FilesystemBackend for offload")

    middleware = SummarizationMiddleware(
        model=model,
        backend=offload_backend,
        keep=defaults["keep"],
        trim_tokens_to_summarize=None,
    )

    # Rebuild the message list the model would see, accounting for
    # any prior offload
    effective = middleware._apply_event_to_messages(
        _normalize_messages(messages), prior_event
    )
    cutoff = middleware._determine_cutoff_index(effective)
    budget_str = format_offload_limit(defaults["keep"], context_limit)

    if cutoff == 0:
        return OffloadThresholdNotMet(
            conversation_tokens=count_tokens_approximately(effective),
            total_context_tokens=total_context_tokens,
            context_limit=context_limit,
            budget_str=budget_str,
        )

    to_summarize, to_keep = middleware._partition_messages(effective, cutoff)

    tokens_summarized = count_tokens_approximately(to_summarize)
    tokens_kept = count_tokens_approximately(to_keep)
    tokens_before = tokens_summarized + tokens_kept

    # Generate summary first so no side effects occur if the LLM fails
    summary = await middleware._acreate_summary(to_summarize)

    backend_path = await offload_messages_to_backend(
        to_summarize,
        middleware,
        thread_id=thread_id,
        backend=offload_backend,
    )
    offload_warning: str | None = None
    if backend_path is None:
        offload_warning = (
            "Warning: conversation history could not be saved to "
            "storage. Older messages will not be recoverable. "
            "Check logs for details."
        )
        logger.error(
            "Backend write failed for thread %s; offloading will proceed "
            "but messages are not recoverable",
            thread_id,
        )
    file_path = backend_path or None

    summary_msg = _coerce_summary_message(
        middleware._build_new_messages_with_path(summary, file_path)[0]
    )

    # Append token savings note so the model is aware of how much context
    # was reclaimed.
    tokens_summary = count_tokens_approximately([summary_msg])
    tokens_after = tokens_summary + tokens_kept
    pct = (
        round((tokens_before - tokens_after) / tokens_before * 100)
        if tokens_before > 0
        else 0
    )
    summarized_before = format_token_count(tokens_summarized)
    summarized_after = format_token_count(tokens_summary)
    savings_note = (
        f"\n\n{len(to_summarize)} messages were offloaded "
        f"({summarized_before} \u2192 {summarized_after} tokens). "
        f"Total context: {format_token_count(tokens_before)} \u2192 "
        f"{format_token_count(tokens_after)} tokens "
        f"({pct}% decrease), "
        f"{len(to_keep)} messages unchanged."
    )
    summary_msg.content += savings_note

    state_cutoff = middleware._compute_state_cutoff(prior_event, cutoff)

    new_event: SummarizationEvent = {
        "cutoff_index": state_cutoff,
        "summary_message": summary_msg,  # ty: ignore[invalid-argument-type]
        "file_path": file_path,
    }

    return OffloadResult(
        new_event=new_event,
        messages_offloaded=len(to_summarize),
        messages_kept=len(to_keep),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        pct_decrease=pct,
        offload_warning=offload_warning,
    )
