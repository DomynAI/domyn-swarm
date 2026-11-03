# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pandas as pd
import pytest

from domyn_swarm.jobs.api.chat_completion import (
    ChatCompletionJob,
    ChatCompletionPerplexityJob,
    CompletionJob,
    MultiChatCompletionJob,
    MultiTurnChatCompletionJob,
    MultiTurnTranslationJob,
    _assistant_message_dict,
    _extract_reasoning_content,
)


@pytest.mark.asyncio
async def test_completion_job(monkeypatch, tmp_path):
    df = pd.DataFrame({"prompt": ["Hello", "World"]})

    mock_resp = AsyncMock()
    mock_resp.choices = [type("C", (), {"text": "Mocked completion"})()]

    job = CompletionJob(model="gpt-3")
    job.client = AsyncMock()
    job.client.completions.create = AsyncMock(return_value=mock_resp)

    results = await job.run(df, tag="test", checkpoint_dir=tmp_path / "checkpoints")

    assert len(results) == 2
    print(results)
    assert results["completion"][0] == "Mocked completion"


@pytest.mark.asyncio
async def test_chat_completion_job(monkeypatch, tmp_path):
    df = pd.DataFrame({"messages": [[{"role": "user", "content": "Hi"}]]})

    mock_choice = type("Choice", (), {"message": type("Msg", (), {"content": "Mocked answer"})()})
    mock_resp = AsyncMock()
    mock_resp.choices = [mock_choice]

    job = ChatCompletionJob(model="gpt-4")
    job.client = AsyncMock()
    job.client.chat.completions.create = AsyncMock(return_value=mock_resp)

    results = await job.run(df, tag="test", checkpoint_dir=tmp_path / "checkpoints")

    assert len(results) == 1
    assert results["result"][0] == "Mocked answer"


@pytest.mark.asyncio
async def test_chat_completion_job_parse_reasoning(monkeypatch, tmp_path):
    df = pd.DataFrame({"messages": [[{"role": "user", "content": "Hi"}]]})
    mock_choice_reasoning = type(
        "Choice",
        (),
        {
            "message": type(
                "Msg",
                (),
                {"content": "Mocked answer", "reasoning_content": "Because I am a bot"},
            )()
        },
    )
    mock_resp_reasoning = AsyncMock()
    mock_resp_reasoning.choices = [mock_choice_reasoning]

    job_reasoning = ChatCompletionJob(model="gpt-4", parse_reasoning=True)
    job_reasoning.client = AsyncMock()
    job_reasoning.client.chat.completions.create = AsyncMock(return_value=mock_resp_reasoning)

    assert job_reasoning.output_cols == ["result", "reasoning_content"]
    results_reasoning = await job_reasoning.run(
        df, tag="test-reasoning", checkpoint_dir=tmp_path / "checkpoints"
    )
    print(results_reasoning)
    assert len(results_reasoning) == 1
    assert results_reasoning["result"][0] == "Mocked answer"
    assert results_reasoning["reasoning_content"][0] == "Because I am a bot"


@pytest.mark.asyncio
async def test_chat_completion_job_parse_reasoning_missing_reasoning_content(monkeypatch, tmp_path):
    df = pd.DataFrame({"messages": [[{"role": "user", "content": "Hi"}]]})
    mock_choice = type("Choice", (), {"message": type("Msg", (), {"content": "Mocked answer"})()})
    mock_resp = AsyncMock()
    mock_resp.choices = [mock_choice]

    job = ChatCompletionJob(model="gpt-4", parse_reasoning=True)
    job.client = AsyncMock()
    job.client.chat.completions.create = AsyncMock(return_value=mock_resp)

    results = await job.run(
        df, tag="test-reasoning-missing", checkpoint_dir=tmp_path / "checkpoints"
    )

    assert len(results) == 1
    assert results["result"][0] == "Mocked answer"
    assert results["reasoning_content"][0] is None


@pytest.mark.asyncio
async def test_multi_chat_completion_job(monkeypatch, tmp_path):
    df = pd.DataFrame({"messages": [[{"role": "user", "content": "Hi"}]]})

    mock_choices = [
        type("Choice", (), {"message": type("Msg", (), {"content": f"Mock {i}"})()})
        for i in range(3)
    ]
    mock_resp = AsyncMock()
    mock_resp.choices = mock_choices

    job = MultiChatCompletionJob(model="gpt-4", n=3, output_cols="generated")
    job.client = AsyncMock()
    job.client.chat.completions.create = AsyncMock(return_value=mock_resp)

    _ = await job.run(df, tag="test", checkpoint_dir=tmp_path / "checkpoints")

    print(job.results)
    assert job.results["generated_1"][0] == "Mock 0"
    assert job.results["generated_2"][0] == "Mock 1"
    assert job.results["generated_3"][0] == "Mock 2"


@pytest.mark.asyncio
async def test_chat_completion_perplexity_job(monkeypatch, tmp_path):
    df = pd.DataFrame({"messages": [[{"role": "user", "content": "Hi"}]]})

    mock_choice = type(
        "Choice",
        (),
        {
            "message": type("Msg", (), {"content": "text!"})(),
            "logprobs": {"tokens": [], "token_logprobs": []},
        },
    )
    mock_resp = AsyncMock()
    mock_resp.choices = [mock_choice]

    monkeypatch.setattr(
        "domyn_swarm.jobs.api.chat_completion.extract_token_logprobs",
        lambda c: [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(
        "domyn_swarm.jobs.api.chat_completion.compute_perplexity_metrics",
        lambda t: (42.0, 21.0),
    )

    job = ChatCompletionPerplexityJob(model="gpt-4")
    job.client = AsyncMock()
    job.client.chat.completions.create = AsyncMock(return_value=mock_resp)

    _ = await job.run(df, tag="test", checkpoint_dir=tmp_path / "checkpoints")

    print(job.results.columns)
    assert job.results["text"][0] == "text!"
    assert job.results["perplexity"][0] == 42.0
    assert job.results["bottom50_perplexity"][0] == 21.0


@pytest.mark.asyncio
async def test_multi_turn_chat_completion_job(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "messages": [
                [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "Tell me a joke"},
                ]
            ]
        }
    )

    mock_resp = AsyncMock()
    mock_resp.choices = [type("C", (), {"message": type("M", (), {"content": "Mocked reply"})()})]

    job = MultiTurnChatCompletionJob(model="gpt-4")
    job.client = AsyncMock()
    job.client.chat.completions.create = AsyncMock(return_value=mock_resp)

    _ = await job.run(df, tag="test", checkpoint_dir=tmp_path / "checkpoints")

    print(job.results)
    results = list(job.results["results"][0])
    assert len(results) == 5
    assert results[-1]["role"] == "assistant"
    assert results[-1]["content"] == "Mocked reply"


@pytest.mark.asyncio
async def test_multi_turn_chat_completion_job_includes_reasoning_content(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "messages": [
                [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Why?"},
                ]
            ]
        }
    )

    mock_msg = type("M", (), {"content": "Mocked reply", "reasoning_content": "Because."})()
    mock_resp = AsyncMock()
    mock_resp.choices = [type("C", (), {"message": mock_msg})]

    job = MultiTurnChatCompletionJob(model="gpt-4")
    job.client = AsyncMock()
    job.client.chat.completions.create = AsyncMock(return_value=mock_resp)

    _ = await job.run(df, tag="test", checkpoint_dir=tmp_path / "checkpoints")

    results = list(job.results["results"][0])
    assert results[-1]["role"] == "assistant"
    assert results[-1]["content"] == "Mocked reply"
    assert results[-1]["reasoning_content"] == "Because."
    assert all(m.get("reasoning_content") is None for m in results[:-1])


@pytest.mark.asyncio
async def test_multi_turn_job_sends_one_session_per_conversation(tmp_path):
    turns = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
        {"role": "user", "content": "e"},
    ]
    df = pd.DataFrame({"messages": [turns, turns]})
    mock_resp = AsyncMock()
    mock_resp.choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})]

    job = MultiTurnChatCompletionJob(model="gpt-4", max_concurrency=1)
    job.client = AsyncMock()
    create = AsyncMock(return_value=mock_resp)
    job.client.chat.completions.create = create

    await job.run(df, tag="test", checkpoint_dir=tmp_path / "checkpoints")

    calls = create.call_args_list
    assert len(calls) == 6
    ids = [c.kwargs["extra_headers"]["X-Session-ID"] for c in calls]
    assert len(set(ids[:3])) == 1 and len(set(ids[3:])) == 1 and ids[0] != ids[3]
    finals = [c.kwargs["extra_headers"].get("X-Dynamo-Session-Final") for c in calls]
    assert finals == [None, None, "true", None, None, "true"]
    assert all(
        c.kwargs["extra_body"]["session_params"]["session_id"]
        == c.kwargs["extra_headers"]["X-Session-ID"]
        for c in calls
    )


@pytest.mark.asyncio
async def test_multi_turn_chat_completion_job_requires_user_or_tool_message(monkeypatch):
    job = MultiTurnChatCompletionJob(model="gpt-4")
    job.client = AsyncMock()

    with pytest.raises(ValueError, match="at least one 'user' or 'tool' message"):
        await job._run_multi_turn([{"role": "system", "content": "Only system"}])


@pytest.mark.asyncio
async def test_multi_turn_translation_job(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "messages": [
                [
                    {"role": "system", "content": "Translate to Italian"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "How are you?"},
                ]
            ]
        }
    )

    mock_resps = [
        AsyncMock(choices=[type("C1", (), {"message": type("M1", (), {"content": "Ciao"})()})]),
        AsyncMock(
            choices=[
                type(
                    "C2",
                    (),
                    {
                        "message": type(
                            "M2", (), {"content": "Come stai?", "reasoning_content": "Direct."}
                        )()
                    },
                )
            ]
        ),
    ]

    job = MultiTurnTranslationJob(model="gpt-4")
    job.client = AsyncMock()
    job.client.chat.completions.create = AsyncMock(side_effect=mock_resps)

    _ = await job.run(df, tag="test", checkpoint_dir=tmp_path / "checkpoints")
    results = list(job.results["results"][0])

    assert results[0]["role"] == "user"
    assert results[0]["content"] == "Ciao"
    assert results[0].get("reasoning_content") is None
    assert results[1]["role"] == "assistant"
    assert results[1]["content"] == "Come stai?"
    assert results[1]["reasoning_content"] == "Direct."

    job.client.chat.completions.create.reset_mock()
    job.client.chat.completions.create.side_effect = list(mock_resps)
    out = await job._run_translation(df.loc[0, "messages"])
    assert out[0]["role"] == results[0]["role"]
    assert out[0]["content"] == results[0]["content"]
    assert out[0].get("reasoning_content") is None
    assert out[1] == {"role": "assistant", "content": "Come stai?", "reasoning_content": "Direct."}


def test_extract_reasoning_content_helpers():
    assert _extract_reasoning_content({"reasoning_content": "x"}) == "x"
    assert _extract_reasoning_content({"other": 1}) is None

    msg = type("Msg", (), {"reasoning_content": "y"})()
    assert _extract_reasoning_content(msg) == "y"

    assert _assistant_message_dict(role="assistant", content="a", message=msg) == {
        "role": "assistant",
        "content": "a",
        "reasoning_content": "y",
    }
    assert _assistant_message_dict(role="assistant", content="a", message=None) == {
        "role": "assistant",
        "content": "a",
    }
