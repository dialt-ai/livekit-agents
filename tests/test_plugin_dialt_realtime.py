from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
from dialt import DialtError, DialtSession, SessionEvent

from livekit import rtc
from livekit.agents import llm
from livekit.plugins import dialt

pytestmark = pytest.mark.unit


class FakeDialtSession:
    def __init__(self) -> None:
        self.events_queue: asyncio.Queue[SessionEvent | None] = asyncio.Queue()
        self.audio: list[bytes] = []
        self.injections: list[dict[str, Any]] = []
        self.instructions: list[str] = []
        self.tool_sets: list[list[dict[str, Any]]] = []
        self.tool_choices: list[tuple[str | dict[str, Any], bool]] = []
        self.tool_results: list[dict[str, Any]] = []
        self.client_events: list[tuple[str, dict[str, Any]]] = []
        self.inject_ack: dict[str, Any] = {"accepted": True, "queued": False}
        self.closed = False

    async def events(self) -> AsyncIterator[SessionEvent]:
        while (event := await self.events_queue.get()) is not None:
            yield event

    async def send_audio(self, audio: bytes) -> None:
        self.audio.append(audio)

    async def inject_context(self, text: str, **kwargs: Any) -> dict[str, Any]:
        self.injections.append({"text": text, **kwargs})
        return self.inject_ack

    async def set_instructions(self, instructions: str, **kwargs: Any) -> None:
        self.instructions.append(instructions)

    async def set_tools(self, tools: list[dict[str, Any]]) -> None:
        self.tool_sets.append(tools)

    async def set_tool_choice(
        self, choice: str | dict[str, Any], *, one_shot: bool = False
    ) -> None:
        self.tool_choices.append((choice, one_shot))

    async def send_tool_result(
        self,
        tool_id: str,
        content: Any,
        *,
        outcome: str,
        verified: bool,
    ) -> None:
        self.tool_results.append(
            {
                "id": tool_id,
                "content": content,
                "outcome": outcome,
                "verified": verified,
            }
        )

    async def send_client_event(self, event: str, **fields: Any) -> None:
        self.client_events.append((event, fields))

    async def finish_input_turn(self) -> None:
        self.client_events.append(("input_audio_done", {}))

    async def close(self) -> None:
        self.closed = True
        await self.events_queue.put(None)

    async def emit(self, event_type: str, **data: Any) -> None:
        await self.events_queue.put(SessionEvent(type=event_type, t_ms=1.0, data=data))


async def wait_for(predicate: Any) -> None:
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0)


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> FakeDialtSession:
    fake = FakeDialtSession()
    monkeypatch.setattr(DialtSession, "connect", AsyncMock(return_value=fake))
    return fake


def make_model(**kwargs: Any) -> dialt.RealtimeModel:
    return dialt.RealtimeModel(
        instructions="You help callers resolve account questions.",
        api_key="test-key",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_connect_uses_sdk_mode_and_keeps_serving_names_private(
    fake_sdk: FakeDialtSession,
) -> None:
    model = make_model(greeting="Welcome", voice="chime", auto_reconnect=True)
    session = model.session(turn_detection_disabled=True)
    await wait_for(lambda: session._sdk_fut.done())

    connect = DialtSession.connect
    kwargs = connect.await_args.kwargs  # type: ignore[attr-defined]
    mode = kwargs["mode"]
    assert mode.instructions == "You help callers resolve account questions."
    assert mode.greeting == "Welcome"
    assert mode.voice == "chime"
    assert mode.turn_detection == "client"
    assert kwargs["auto_reconnect"] is True
    assert model.model == "unknown"
    assert model.provider == "unknown"

    @llm.function_tool(name="lookup")
    async def lookup(account: str) -> str:
        return account

    await session.update_instructions("You now help with billing questions.")
    await session.update_tools([lookup])
    assert fake_sdk.instructions == ["You now help with billing questions."]
    assert fake_sdk.tool_sets[-1][0]["name"] == "lookup"

    session.push_audio(
        rtc.AudioFrame(
            data=np.zeros(160, dtype=np.int16).tobytes(),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=160,
        )
    )
    await wait_for(lambda: bool(fake_sdk.audio))
    assert len(fake_sdk.audio[0]) == 320

    session.commit_audio()
    await wait_for(lambda: fake_sdk.client_events == [("input_audio_done", {})])
    await session.aclose()


@pytest.mark.asyncio
async def test_generation_tool_call_interruption_and_playback_reporting(
    fake_sdk: FakeDialtSession,
) -> None:
    session = make_model().session()
    reply = session.generate_reply(
        instructions="Answer with the current balance.",
        tool_choice={"type": "function", "function": {"name": "lookup_balance"}},
    )
    await wait_for(lambda: bool(fake_sdk.injections))
    assert fake_sdk.injections[0]["text"] == "Answer with the current balance."
    assert fake_sdk.injections[0]["reply"] is True
    assert fake_sdk.tool_choices == [({"tool": "lookup_balance"}, True)]

    await fake_sdk.emit("turn", turn_id="turn-1")
    generation = await asyncio.wait_for(reply, 1)
    assert generation.user_initiated is True
    message = await anext(generation.message_stream.__aiter__())

    await fake_sdk.emit("text_delta", turn_id="turn-1", delta="Your balance is ")
    await fake_sdk.events_queue.put(
        SessionEvent(
            type="audio",
            t_ms=2.0,
            audio=np.full(1600, 0.25, dtype=np.float32),
        )
    )
    await fake_sdk.emit("tool_call", id="call-1", name="lookup_balance", args={"account": "A-7"})

    assert await anext(message.text_stream.__aiter__()) == "Your balance is "
    audio = await anext(message.audio_stream.__aiter__())
    assert audio.sample_rate == 16000
    call = await anext(generation.function_stream.__aiter__())
    assert call.call_id == "call-1"
    assert json.loads(call.arguments) == {"account": "A-7"}

    speech_started: list[object] = []
    session.on("input_speech_started", speech_started.append)
    await fake_sdk.emit("interrupted", turn_id="turn-1", barge_seq=9, clear=True)
    await wait_for(lambda: bool(speech_started))
    session.truncate(message_id="turn-1", modalities=["audio", "text"], audio_end_ms=25)
    await wait_for(lambda: bool(fake_sdk.client_events))
    assert fake_sdk.client_events[-1] == (
        "playback_stopped",
        {"discarded_ms": 75, "barge_seq": 9},
    )
    await session.aclose()


@pytest.mark.asyncio
async def test_context_mutations_and_tool_results_preserve_roles_and_failures(
    fake_sdk: FakeDialtSession,
) -> None:
    session = make_model().session()
    await wait_for(lambda: session._sdk_fut.done())
    context = llm.ChatContext.empty()
    context.add_message(role="developer", content="The caller is on the premium plan.", id="ctx")
    context.add_message(role="user", content="What is my balance?", id="user")
    context.add_message(role="assistant", content="I will check.", id="assistant")
    context.items.append(
        llm.FunctionCallOutput(
            id="result",
            call_id="call-2",
            name="lookup_balance",
            output="backend unavailable",
            is_error=True,
        )
    )

    await session.update_chat_ctx(context)

    assert fake_sdk.injections == [
        {
            "text": "The caller is on the premium plan.",
            "role": "context",
            "reply": False,
            "message_id": "ctx",
        },
        {
            "text": "What is my balance?",
            "role": "user",
            "reply": False,
            "message_id": "user",
        },
    ]
    assert fake_sdk.tool_results == [
        {
            "id": "call-2",
            "content": "backend unavailable",
            "outcome": "failed",
            "verified": False,
        }
    ]
    await session.aclose()


@pytest.mark.asyncio
async def test_rejected_reply_machine_error_and_reconnect_are_asymmetric(
    fake_sdk: FakeDialtSession,
) -> None:
    session = make_model().session()
    fake_sdk.inject_ack = {
        "accepted": False,
        "code": "reply_busy",
        "detail": "reply is already running",
        "retryable": True,
    }
    reply = session.generate_reply()
    with pytest.raises(llm.RealtimeError, match="already running") as raised:
        await reply
    assert raised.value.code == "reply_busy"

    errors: list[llm.RealtimeModelError] = []
    reconnected: list[object] = []
    transcripts: list[llm.InputTranscriptionCompleted] = []
    session.on("error", errors.append)
    session.on("session_reconnected", reconnected.append)
    session.on("input_audio_transcription_completed", transcripts.append)
    await fake_sdk.emit("asr", turn_id="input-1", text="I need a flight")
    await fake_sdk.emit("asr_correction", turn_id="input-1", text="I need a refund")
    await fake_sdk.emit("error", code="server_busy", detail="busy", retryable=True)
    await fake_sdk.emit("resume_failed", code="resume_failed", detail="expired", retryable=False)
    await fake_sdk.emit("reconnected", attempt=2)
    await wait_for(lambda: len(errors) == 2 and len(reconnected) == 1)
    assert errors[0].recoverable is True
    assert cast_realtime_error(errors[0]).code == "server_busy"
    assert errors[1].recoverable is False
    assert cast_realtime_error(errors[1]).code == "resume_failed"
    await wait_for(lambda: len(transcripts) == 2)
    corrected = session.chat_ctx.get_by_id("input-1")
    assert isinstance(corrected, llm.ChatMessage)
    assert corrected.text_content == "I need a refund"
    await session.aclose()


def cast_realtime_error(error: llm.RealtimeModelError) -> llm.RealtimeError:
    assert isinstance(error.error, llm.RealtimeError)
    return error.error


def test_tool_options_exports_and_package_metadata() -> None:
    @llm.function_tool(name="lookup", description="Look up a record")
    async def lookup(record_id: str) -> str:
        return record_id

    definitions = dialt.utils.to_dialt_tools(
        [lookup],
        {
            "lookup": dialt.ToolOptions(
                read_only=True,
                requires_permission=False,
                permission_source="application",
                resolver_only=True,
                expected_duration="seconds",
                status_label="Looking up the record",
                timeout=12.5,
                deferred=True,
                deferred_timeout=90,
                notify_on_complete=True,
            )
        },
    )
    assert definitions[0] == {
        "name": "lookup",
        "description": "Look up a record",
        "parameters": definitions[0]["parameters"],
        "read_only": True,
        "requires_permission": False,
        "permission_source": "application",
        "resolver_only": True,
        "expected_duration": "seconds",
        "status_label": "Looking up the record",
        "timeout": 12.5,
        "deferred": True,
        "deferred_timeout": 90,
        "notify_on_complete": True,
    }
    assert definitions[0]["parameters"]["properties"]["record_id"]["type"] == "string"
    assert definitions[0]["parameters"]["required"] == ["record_id"]

    pyproject = Path(__file__).parents[1] / "livekit-plugins/livekit-plugins-dialt/pyproject.toml"
    text = pyproject.read_text()
    assert 'name = "livekit-plugins-dialt"' in text
    assert '"dialt-sdk==0.41.0"' in text
    assert dialt.RealtimeModel is dialt.realtime.RealtimeModel


def test_local_validation_and_sdk_error_code_are_not_flattened() -> None:
    with pytest.raises(ValueError, match="instructions are required"):
        dialt.RealtimeModel(instructions=" ", api_key="test-key")

    error = DialtError("try later", code="server_busy", retryable=True)
    assert error.code == "server_busy"
    assert error.retryable is True
