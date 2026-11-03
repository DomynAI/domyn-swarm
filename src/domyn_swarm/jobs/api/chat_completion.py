# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion, Choice
from pydantic import model_validator

from domyn_swarm.helpers.data import compute_perplexity_metrics, extract_token_logprobs
from domyn_swarm.jobs.api.base import SwarmJob
from domyn_swarm.jobs.api.config import JobConfig
from domyn_swarm.jobs.api.session import new_session_id, session_request_options


def _extract_reasoning_content(message: Any) -> Any | None:
    if message is None:
        return None
    if isinstance(message, Mapping):
        return message.get("reasoning_content")
    return getattr(message, "reasoning_content", None)


def _assistant_message_dict(*, role: str, content: Any, message: Any) -> dict[str, Any]:
    response_dict: dict[str, Any] = {"role": role, "content": content}
    reasoning_content = _extract_reasoning_content(message)
    if reasoning_content is not None:
        response_dict["reasoning_content"] = reasoning_content
    return response_dict


class ChatCompletionConfig(JobConfig):
    """Configuration for `ChatCompletionJob`.

    Attributes:
        parse_reasoning: Whether to split the provider's reasoning content into
            a second output column.
    """

    parse_reasoning: bool = False

    @model_validator(mode="after")
    def _apply_parse_reasoning(self) -> "ChatCompletionConfig":
        """Derive the output columns from `parse_reasoning`.

        Returns:
            This config, with `output_cols` and `default_output_cols` set.
        """
        cols = ["result", "reasoning_content"] if self.parse_reasoning else "result"
        object.__setattr__(self, "output_cols", cols)
        object.__setattr__(self, "default_output_cols", cols if isinstance(cols, list) else [cols])
        return self


class MultiChatCompletionConfig(JobConfig):
    """Configuration for `MultiChatCompletionJob`.

    Attributes:
        n: Number of independent completions produced per row.
    """

    n: int = 3

    @model_validator(mode="after")
    def _expand_output_cols(self) -> "MultiChatCompletionConfig":
        """Expand a base output column name into `n` numbered columns.

        Returns:
            This config, with `output_cols` and `default_output_cols` set.
        """
        base = self.output_cols
        if isinstance(base, str):
            cols = [f"{base}_{i + 1}" for i in range(self.n)]
            object.__setattr__(self, "output_cols", cols)
            object.__setattr__(self, "default_output_cols", cols)
        return self


class CompletionJob(SwarmJob):
    """
    Input DF must have column `prompt`; output DF gets `completion`.
    """

    api_version = 2

    def __init__(
        self,
        *,
        endpoint=None,
        model="",
        input_column_name="prompt",
        output_cols="completion",
        checkpoint_interval=16,
        max_concurrency=2,
        retries=5,
        timeout=600,
        **extra_kwargs,
    ):
        super().__init__(
            endpoint=endpoint,
            model=model,
            input_column_name=input_column_name,
            output_cols=output_cols,
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
            retries=retries,
            timeout=timeout,
            **extra_kwargs,
        )

    async def transform_items(self, items: list[str]) -> list[Any]:
        """
        Transform a list of prompts into a list of completions.
        Each item in `items` is a single prompt string.

        Args:
            items: List of prompt strings.

        Returns a list of completion strings.
        """
        outs = []
        extra_body = self._request_kwargs()
        for prompt in items:
            resp = await self.client.completions.create(
                model=self.model, prompt=prompt, extra_body=extra_body
            )
            outs.append(resp.choices[0].text)
        return outs


class ChatCompletionJob(SwarmJob):
    """Input DF must have column `messages` (list of dicts).

    Output DF gets `result`, plus `reasoning_content` when `parse_reasoning` is
    set.
    """

    api_version = 2
    config_class = ChatCompletionConfig

    async def transform_items(self, items: list[list[ChatCompletionMessageParam]]) -> list[Any]:
        outs = []
        extra_body = self._request_kwargs()
        for msgs in items:
            resp = await self.client.chat.completions.create(
                model=self.model, messages=msgs, extra_body=extra_body
            )
            choice = resp.choices[0]
            if self.parse_reasoning:
                outs.append(
                    (
                        choice.message.content,
                        _extract_reasoning_content(choice.message),
                    )
                )
            else:
                outs.append(choice.message.content)
        return outs


class MultiChatCompletionJob(SwarmJob):
    """Produce *n* independent chat completions for every row.

    Input column: `messages`. Output columns: `result_1` … `result_n`, or the
    supplied `output_cols` base name numbered the same way.
    """

    api_version = 2
    config_class = MultiChatCompletionConfig

    async def transform_items(self, items: list[list[ChatCompletionMessageParam]]) -> list[Any]:
        outs = []
        extra_body = self._request_kwargs()
        for msgs in items:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                n=self.n,  # ask the API for n choices at once
                extra_body=extra_body,
            )
            outs.append([choice.message.content for choice in resp.choices])
        return outs


class PerplexityMixin:
    def compute_from_choice(self, choice: Choice) -> tuple[float, float]:
        token_logprobs = extract_token_logprobs(choice)
        perp, bottom50 = compute_perplexity_metrics(token_logprobs)
        return perp, bottom50


class ChatCompletionPerplexityJob(PerplexityMixin, SwarmJob):
    """
    For each row's `messages` (a list of dicts), produce a single chat
    completion with logprobs, and compute perplexity metrics.

    - Input  column: `messages`
    - Output columns: `text`, `perplexity`, `bottom50_perplexity`
    """

    api_version = 2

    def __init__(
        self,
        *,
        endpoint=None,
        model="",
        input_column_name="messages",
        output_cols="result",
        checkpoint_interval=16,
        max_concurrency=2,
        retries=5,
        **extra_kwargs,
    ):
        super().__init__(
            endpoint=endpoint,
            model=model,
            input_column_name=input_column_name,
            output_cols=["text", "perplexity", "bottom50_perplexity"],
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
            retries=retries,
            **extra_kwargs,
        )

    async def transform_items(self, items: list[list[ChatCompletionMessageParam]]) -> list[Any]:
        outs = []
        extra_body = self._request_kwargs()
        for msgs in items:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                logprobs=True,
                extra_body=extra_body,
            )
            choice = resp.choices[0]
            text = choice.message.content
            perp, bottom50 = self.compute_from_choice(choice)
            outs.append((text, perp, bottom50))
        return outs


class MultiTurnChatCompletionJob(SwarmJob):
    """
    For each row's `messages` (a list of dicts), replay the conversation
    turn by turn, appending the assistant's reply after each user/tool message.

    - Input  column: `messages`
    - Output column: `running_messages`
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str = "",
        input_column_name: str = "messages",
        output_cols: str = "results",
        checkpoint_interval: int = 16,
        max_concurrency: int = 2,
        retries: int = 5,
        **extra_kwargs: Any,
    ):
        super().__init__(
            endpoint=endpoint,
            model=model,
            input_column_name=input_column_name,
            output_cols=output_cols,
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
            retries=retries,
            **extra_kwargs,
        )

    async def _run_multi_turn(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        extra_body: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if extra_body is None:
            extra_body = self._request_kwargs()
        # Find indices of user or tool messages
        user_idxs = [i for i, m in enumerate(messages) if m["role"] in {"user", "tool"}]
        if not user_idxs:
            raise ValueError("Input must contain at least one 'user' or 'tool' message")

        running_request: list[ChatCompletionMessageParam] = []
        reasoning_by_index: list[Any | None] = []
        session_id = new_session_id()

        # Zip together slice starts and ends
        idx = 0

        for i in user_idxs:
            session_options = session_request_options(
                session_id, extra_body, final=i == user_idxs[-1]
            )
            # append all the messages until the next user message (including system, etc.)
            running_request.extend(messages[idx : i + 1])
            reasoning_by_index.extend([None] * len(messages[idx : i + 1]))

            resp: ChatCompletion = await self.client.chat.completions.create(
                model=self.model, messages=running_request, **session_options
            )
            choice = resp.choices[0]

            # append the assistant's response to the messages
            running_request.append(
                ChatCompletionAssistantMessageParam(
                    {"role": "assistant", "content": choice.message.content}
                )
            )
            reasoning_by_index.append(_extract_reasoning_content(choice.message))

            # update the index skipping assistant message
            idx = i + 2
        output: list[dict[str, Any]] = []
        for msg, reasoning in zip(running_request, reasoning_by_index, strict=True):
            msg_dict = dict(msg)
            if reasoning is not None and msg_dict.get("role") == "assistant":
                msg_dict["reasoning_content"] = reasoning
            output.append(msg_dict)
        return output

    async def transform_items(self, items: list[list[ChatCompletionMessageParam]]) -> list[Any]:
        outs = []
        extra_body = self._request_kwargs()

        outs = [await self._run_multi_turn(msgs, extra_body=extra_body) for msgs in items]
        return outs


class MultiTurnTranslationJob(SwarmJob):
    """Translate each non-system message while preserving its original role.

    The first message supplies the translation system prompt and is prepended
    to each request. The input column is `messages`; the output column is
    `results`.
    """

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str = "",
        input_column_name: str = "messages",
        output_cols: str = "results",
        checkpoint_interval: int = 16,
        max_concurrency: int = 2,
        retries: int = 5,
        **extra_kwargs: Any,
    ):
        super().__init__(
            endpoint=endpoint,
            model=model,
            input_column_name=input_column_name,
            output_cols=output_cols,
            checkpoint_interval=checkpoint_interval,
            max_concurrency=max_concurrency,
            retries=retries,
            **extra_kwargs,
        )

    async def _run_translation(
        self,
        messages: list[dict[str, Any]],
        *,
        extra_body: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if extra_body is None:
            extra_body = self._request_kwargs()
        running: list[dict[str, Any]] = []

        for i in range(1, len(messages)):
            query = [
                messages[0],
                {
                    "role": "user",
                    "content": messages[i]["content"],
                },
            ]

            resp: ChatCompletion = await self.client.chat.completions.create(
                model=self.model, messages=query, extra_body=extra_body
            )
            choice = resp.choices[0]

            running.append(
                _assistant_message_dict(
                    role=messages[i]["role"], content=choice.message.content, message=choice.message
                )
            )

        return running

    async def transform_items(self, items: list[list[dict[str, Any]]]) -> list[Any]:
        extra_body = self._request_kwargs()
        return [await self._run_translation(messages, extra_body=extra_body) for messages in items]
