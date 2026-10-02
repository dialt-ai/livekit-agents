from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from dialt import (
    DEFAULT_REALTIME_URL,
    DEFAULT_SR,
    OUTPUT_SR,
    DialtError,
    DialtMode,
    DialtSession,
    SessionEvent,
    float32_to_pcm16,
)
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.llm.utils import compute_chat_ctx_diff
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ..log import logger
from ..models import ToolOptions
from ..utils import to_dialt_tools

NUM_CHANNELS = 1


@dataclass
class _Options:
    api_key: str
    url: str
    instructions: str
    voice: str | None
    greeting: str | bool | None
    web_search: bool
    end_call: bool
    end_call_when: str | None
    brain: Literal["fast", "smart"]
    silence_nudge_s: float | None
    silence_end_s: float | None
    tool_choice: llm.ToolChoice | None
    policy: dict[str, Any] | None
    user: str | None
    timezone: str | None
    session_id: str | None
    tool_options: dict[str, ToolOptions]
    connect_timeout_s: float
    auto_reconnect: bool
    reconnect_base_s: float
    reconnect_max_s: float
    max_reconnect_attempts: int


@dataclass
class _Generation:
    turn_id: str
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    output_text: str = ""
    generated_audio_ms: int = 0
    barge_seq: int | None = None
    done: bool = False


@dataclass
class _PendingReply:
    future: asyncio.Future[llm.GenerationCreatedEvent]
    message_id: str
    acknowledged: bool = False
    request_sent: bool = False


def _dialt_tool_choice(choice: llm.ToolChoice | None) -> str | dict[str, Any]:
    if choice is None:
        return "auto"
    if isinstance(choice, str):
        return choice
    return {"tool": choice["function"]["name"]}


class RealtimeModel(llm.RealtimeModel):
    """A LiveKit realtime model whose conversation is owned by Dialt."""

    def __init__(
        self,
        *,
        instructions: str,
        api_key: str | None = None,
        url: str = DEFAULT_REALTIME_URL,
        voice: str | None = None,
        greeting: str | bool | None = None,
        web_search: bool = False,
        end_call: bool = False,
        end_call_when: str | None = None,
        brain: Literal["fast", "smart"] = "fast",
        silence_nudge_s: float | None = None,
        silence_end_s: float | None = None,
        tool_choice: llm.ToolChoice | None = None,
        policy: dict[str, Any] | None = None,
        user: str | None = None,
        timezone: str | None = None,
        session_id: str | None = None,
        tool_options: dict[str, ToolOptions] | None = None,
        connect_timeout_s: float = 15.0,
        auto_reconnect: bool = True,
        reconnect_base_s: float = 0.5,
        reconnect_max_s: float = 5.0,
        max_reconnect_attempts: int = 12,
    ) -> None:
        if not instructions.strip():
            raise ValueError(
                "instructions are required: describe who the agent is and what it does"
            )
        resolved_api_key = api_key or os.getenv("DIALT_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Dialt API key is required. Pass api_key or set the DIALT_API_KEY environment variable."
            )

        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=True,
                audio_output=True,
                manual_function_calls=False,
                can_disable_turn_detection=True,
                supports_overlapping_speech=False,
                mutable_chat_context=False,
                mutable_instructions=True,
                mutable_tools=True,
                per_response_tool_choice=False,
            )
        )
        self._opts = _Options(
            api_key=resolved_api_key,
            url=url,
            instructions=instructions,
            voice=voice,
            greeting=greeting,
            web_search=web_search,
            end_call=end_call,
            end_call_when=end_call_when,
            brain=brain,
            silence_nudge_s=silence_nudge_s,
            silence_end_s=silence_end_s,
            tool_choice=tool_choice,
            policy=policy,
            user=user,
            timezone=timezone,
            session_id=session_id,
            tool_options=tool_options or {},
            connect_timeout_s=connect_timeout_s,
            auto_reconnect=auto_reconnect,
            reconnect_base_s=reconnect_base_s,
            reconnect_max_s=reconnect_max_s,
            max_reconnect_attempts=max_reconnect_attempts,
        )
        self._sessions = weakref.WeakSet[RealtimeSession]()

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        session = RealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        self._sessions.add(session)
        return session

    async def aclose(self) -> None:
        if self._sessions:
            await asyncio.gather(*(session.aclose() for session in list(self._sessions)))


class RealtimeSession(llm.RealtimeSession[Literal["dialt_server_event_received"]]):
    def __init__(
        self, realtime_model: RealtimeModel, *, turn_detection_disabled: bool = False
    ) -> None:
        super().__init__(realtime_model)
        self._realtime_model = realtime_model
        self._opts = replace(
            realtime_model._opts,
            tool_options=dict(realtime_model._opts.tool_options),
        )
        self._turn_detection_disabled = turn_detection_disabled
        self._instructions = self._opts.instructions
        self._tools = llm.ToolContext.empty()
        self._chat_ctx = llm.ChatContext.empty()
        self._chat_ctx_lock = asyncio.Lock()
        self._input_resampler: rtc.AudioResampler | None = None
        self._input_resampler_rate: int | None = None
        self._audio_ch = utils.aio.Chan[bytes]()
        self._sdk_fut = asyncio.Future[DialtSession]()
        self._pending_replies: list[_PendingReply] = []
        self._reply_tasks: set[asyncio.Task[None]] = set()
        self._tool_choice_task: asyncio.Task[None] | None = None
        self._current_generation: _Generation | None = None
        self._recent_generations: dict[str, _Generation] = {}
        self._discarded_turn_id: str | None = None
        self._input_speech_active = False
        self._closed = False
        self._aclose_done = False
        self._main_atask = asyncio.create_task(self._main_task(), name="DialtSession._main_task")

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    async def _sdk_session(self) -> DialtSession:
        try:
            return await asyncio.shield(self._sdk_fut)
        except DialtError as exc:
            raise llm.RealtimeError(str(exc), code=exc.code) from exc

    async def update_instructions(self, instructions: str) -> None:
        if not instructions.strip():
            raise ValueError("instructions must not be blank")
        self._instructions = instructions
        sdk = await self._sdk_session()
        try:
            await sdk.set_instructions(instructions)
        except DialtError as exc:
            raise llm.RealtimeError(str(exc), code=exc.code) from exc

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        self._tools.update_tools(tools)
        sdk = await self._sdk_session()
        definitions = to_dialt_tools(self._tools.flatten(), self._opts.tool_options)
        try:
            await sdk.set_tools(definitions)
            if self._opts.tool_choice is not None:
                await sdk.set_tool_choice(_dialt_tool_choice(self._opts.tool_choice))
        except DialtError as exc:
            raise llm.RealtimeError(str(exc), code=exc.code) from exc

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        async with self._chat_ctx_lock:
            diff = compute_chat_ctx_diff(self._chat_ctx, chat_ctx)
            if diff.to_remove or diff.to_update:
                logger.debug(
                    "Dialt ignores removed or changed chat items",
                    extra={"removed": len(diff.to_remove), "changed": len(diff.to_update)},
                )

            sdk = await self._sdk_session()
            try:
                for _, item_id in diff.to_create:
                    item = chat_ctx.get_by_id(item_id)
                    if isinstance(item, llm.ChatMessage) and item.raw_text_content:
                        if item.role in ("system", "developer"):
                            await sdk.inject_context(
                                item.raw_text_content,
                                role="context",
                                reply=False,
                                message_id=item.id,
                            )
                        elif item.role == "user":
                            await sdk.inject_context(
                                item.raw_text_content,
                                role="user",
                                reply=False,
                                message_id=item.id,
                            )
                    elif isinstance(item, llm.FunctionCallOutput):
                        await sdk.send_tool_result(
                            item.call_id,
                            item.output,
                            outcome="failed" if item.is_error else "succeeded",
                            verified=False,
                        )
            except DialtError as exc:
                raise llm.RealtimeError(str(exc), code=exc.code) from exc
            self._chat_ctx = chat_ctx.copy()

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        if not is_given(tool_choice):
            return
        self._opts.tool_choice = tool_choice
        task = asyncio.create_task(self._set_tool_choice(tool_choice), name="dialt-set-tool-choice")
        self._tool_choice_task = task
        self._reply_tasks.add(task)
        task.add_done_callback(self._reply_tasks.discard)

    async def _set_tool_choice(self, tool_choice: llm.ToolChoice | None) -> None:
        sdk = await self._sdk_session()
        await sdk.set_tool_choice(_dialt_tool_choice(tool_choice))

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        for resampled in self._resample_audio(frame):
            with contextlib.suppress(utils.aio.channel.ChanClosed):
                self._audio_ch.send_nowait(resampled.data.tobytes())

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler is not None and frame.sample_rate != self._input_resampler_rate:
            self._input_resampler = None
            self._input_resampler_rate = None
        if self._input_resampler is None and (
            frame.sample_rate != DEFAULT_SR or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=DEFAULT_SR,
                num_channels=NUM_CHANNELS,
            )
            self._input_resampler_rate = frame.sample_rate
        if self._input_resampler is not None:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def push_video(self, frame: rtc.VideoFrame) -> None:
        del frame
        logger.warning("push_video() is not supported by Dialt")

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        if self._closed:
            fut.set_exception(llm.RealtimeError("Dialt session is closed"))
            return fut
        pending = _PendingReply(future=fut, message_id=utils.shortuuid("dialt_reply_"))
        self._pending_replies.append(pending)
        task = asyncio.create_task(
            self._request_reply(
                pending, instructions=instructions, tool_choice=tool_choice, tools=tools
            ),
            name="dialt-generate-reply",
        )
        self._reply_tasks.add(task)
        task.add_done_callback(self._reply_tasks.discard)

        def remove_pending(_: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            if fut.cancelled():
                task.cancel()
                if not pending.request_sent:
                    self._remove_pending_reply(pending)
            else:
                self._remove_pending_reply(pending)

        fut.add_done_callback(remove_pending)
        return fut

    async def _request_reply(
        self,
        pending: _PendingReply,
        *,
        instructions: NotGivenOr[str],
        tool_choice: NotGivenOr[llm.ToolChoice],
        tools: NotGivenOr[list[llm.Tool]],
    ) -> None:
        try:
            sdk = await self._sdk_session()
            if (
                self._tool_choice_task is not None
                and self._tool_choice_task is not asyncio.current_task()
            ):
                await asyncio.shield(self._tool_choice_task)
            if is_given(tools):
                raise llm.RealtimeError(
                    "Dialt does not support per-response tools; update session tools first"
                )
            if is_given(tool_choice):
                await sdk.set_tool_choice(_dialt_tool_choice(tool_choice), one_shot=True)
            prompt = instructions if is_given(instructions) else "Respond to the caller now."
            pending.request_sent = True
            ack = await sdk.inject_context(
                prompt,
                role="context",
                reply=True,
                message_id=pending.message_id,
            )
            if not ack.get("accepted", False):
                raise llm.RealtimeError(
                    str(ack.get("detail") or "Dialt rejected the reply request"),
                    code=cast(str | None, ack.get("code")),
                )
        except Exception as exc:
            self._remove_pending_reply(pending)
            if not pending.future.done():
                if isinstance(exc, DialtError):
                    exc = llm.RealtimeError(str(exc), code=exc.code)
                pending.future.set_exception(exc)

    def _remove_pending_reply(self, pending: _PendingReply) -> None:
        with contextlib.suppress(ValueError):
            self._pending_replies.remove(pending)

    def commit_audio(self) -> None:
        if not self._turn_detection_disabled or self._closed:
            return
        task = asyncio.create_task(self._finish_input_turn(), name="dialt-finish-input-turn")
        self._reply_tasks.add(task)
        task.add_done_callback(self._reply_tasks.discard)

    async def _finish_input_turn(self) -> None:
        await (await self._sdk_session()).finish_input_turn()

    def clear_audio(self) -> None:
        logger.warning("clear_audio() cannot recall audio already sent to Dialt")

    def interrupt(self) -> None:
        if self._current_generation is not None:
            self._close_generation(self._current_generation, interrupted=True)

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        del modalities
        generation = self._recent_generations.get(message_id)
        if generation is None:
            return
        if is_given(audio_transcript):
            item = self._chat_ctx.get_by_id(message_id)
            if isinstance(item, llm.ChatMessage):
                item.content = [audio_transcript]
                item.interrupted = True
        discarded_ms = max(0, generation.generated_audio_ms - audio_end_ms)
        task = asyncio.create_task(
            self._report_playback_stopped(discarded_ms, generation.barge_seq),
            name="dialt-playback-stopped",
        )
        self._reply_tasks.add(task)
        task.add_done_callback(self._reply_tasks.discard)

    async def _report_playback_stopped(self, discarded_ms: int, barge_seq: int | None) -> None:
        fields: dict[str, Any] = {"discarded_ms": discarded_ms}
        if barge_seq is not None:
            fields["barge_seq"] = barge_seq
        await (await self._sdk_session()).send_client_event("playback_stopped", **fields)

    async def aclose(self) -> None:
        if self._aclose_done:
            return
        self._aclose_done = True
        self._closed = True
        self._audio_ch.close()
        if self._reply_tasks:
            await utils.aio.cancel_and_wait(*self._reply_tasks)
        await utils.aio.cancel_and_wait(self._main_atask)
        if self._sdk_fut.done() and not self._sdk_fut.cancelled():
            with contextlib.suppress(Exception):
                await self._sdk_fut.result().close()
        for pending in self._pending_replies:
            if not pending.future.done():
                pending.future.cancel("Dialt session closed")
        if self._current_generation is not None:
            self._close_generation(self._current_generation, interrupted=True)

    async def _main_task(self) -> None:
        started = time.perf_counter()
        try:
            initial_tools = to_dialt_tools(self._tools.flatten(), self._opts.tool_options)
            mode = DialtMode(
                instructions=self._instructions,
                voice=self._opts.voice,
                tools=initial_tools,
                web_search=self._opts.web_search,
                end_call=self._opts.end_call,
                end_call_when=self._opts.end_call_when,
                greeting=self._opts.greeting,
                turn_detection="client" if self._turn_detection_disabled else "server",
                brain=self._opts.brain,
                silence_nudge_s=self._opts.silence_nudge_s,
                silence_end_s=self._opts.silence_end_s,
                tool_choice=_dialt_tool_choice(self._opts.tool_choice)
                if self._opts.tool_choice is not None and initial_tools
                else None,
                policy=self._opts.policy,
            )
            sdk = await DialtSession.connect(
                self._opts.url,
                api_key=self._opts.api_key,
                mode=mode,
                user=self._opts.user,
                timezone=self._opts.timezone,
                session_id=self._opts.session_id,
                connect_timeout_s=self._opts.connect_timeout_s,
                auto_reconnect=self._opts.auto_reconnect,
                reconnect_base_s=self._opts.reconnect_base_s,
                reconnect_max_s=self._opts.reconnect_max_s,
                max_reconnect_attempts=self._opts.max_reconnect_attempts,
            )
            self._sdk_fut.set_result(sdk)
            self._report_connection_acquired(time.perf_counter() - started)
            event_task = asyncio.create_task(self._event_loop(sdk), name="dialt-events")
            audio_task = asyncio.create_task(self._audio_loop(sdk), name="dialt-audio")
            done, pending = await asyncio.wait(
                (event_task, audio_task), return_when=asyncio.FIRST_COMPLETED
            )
            await utils.aio.cancel_and_wait(*pending)
            for task in done:
                task.result()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._sdk_fut.done():
                self._sdk_fut.set_exception(exc)
            recoverable = bool(getattr(exc, "retryable", False))
            self._emit_error(exc, recoverable=recoverable)
        finally:
            if not self._closed:
                self._terminate_session("Dialt session ended")

    def _terminate_session(self, reason: str) -> None:
        self._closed = True
        self._audio_ch.close()
        error = llm.RealtimeError(reason, code="connection_closed")
        for pending in list(self._pending_replies):
            if not pending.future.done():
                pending.future.set_exception(error)
        self._pending_replies.clear()
        if self._current_generation is not None:
            self._close_generation(self._current_generation, interrupted=True)

    async def _audio_loop(self, sdk: DialtSession) -> None:
        async for audio in self._audio_ch:
            await sdk.send_audio(audio)

    async def _event_loop(self, sdk: DialtSession) -> None:
        async for event in sdk.events():
            self.emit("dialt_server_event_received", event)
            self._handle_event(event)

    def _handle_event(self, event: SessionEvent) -> None:
        data = event.data
        if event.type == "turn":
            turn_id = str(data.get("turn_id") or utils.shortuuid("dialt_turn_"))
            canceled = next(
                (
                    item
                    for item in self._pending_replies
                    if item.acknowledged and item.future.cancelled()
                ),
                None,
            )
            if canceled is not None:
                self._remove_pending_reply(canceled)
                if self._current_generation is not None:
                    self._close_generation(self._current_generation, interrupted=True)
                self._discarded_turn_id = turn_id
            else:
                self._start_generation(turn_id)
        elif event.type == "inject_context_ack":
            message_id = data.get("message_id")
            pending = next(
                (item for item in self._pending_replies if item.message_id == message_id), None
            )
            if pending is not None:
                if data.get("accepted", False):
                    pending.acknowledged = True
                else:
                    self._remove_pending_reply(pending)
        elif event.type == "text_delta":
            generation = self._generation_for(data.get("turn_id"))
            delta = data.get("delta")
            if generation is not None and isinstance(delta, str) and delta:
                generation.output_text += delta
                generation.text_ch.send_nowait(delta)
        elif event.type == "audio" and event.audio is not None:
            generation = self._current_generation
            if generation is not None and not generation.done and event.audio.size:
                pcm = float32_to_pcm16(event.audio)
                samples = len(pcm) // 2
                generation.audio_ch.send_nowait(
                    rtc.AudioFrame(
                        data=pcm,
                        sample_rate=OUTPUT_SR,
                        num_channels=NUM_CHANNELS,
                        samples_per_channel=samples,
                    )
                )
                generation.generated_audio_ms += round(samples * 1000 / OUTPUT_SR)
        elif event.type == "utterance":
            generation = self._generation_for(data.get("turn_id"))
            text = data.get("text")
            if generation is not None and isinstance(text, str) and not generation.output_text:
                generation.output_text = text
                generation.text_ch.send_nowait(text)
        elif event.type == "done":
            if str(data.get("turn_id")) == self._discarded_turn_id:
                self._discarded_turn_id = None
                return
            generation = self._generation_for(data.get("turn_id"))
            if generation is not None:
                self._close_generation(generation, interrupted=False)
        elif event.type == "tool_call":
            if self._discarded_turn_id is None:
                self._handle_tool_call(data)
        elif event.type == "interrupted":
            if str(data.get("turn_id")) == self._discarded_turn_id:
                self._discarded_turn_id = None
                if not self._input_speech_active:
                    self._input_speech_active = True
                    self.emit("input_speech_started", llm.InputSpeechStartedEvent())
                return
            generation = self._generation_for(data.get("turn_id"))
            if generation is not None:
                generation.barge_seq = cast(int | None, data.get("barge_seq"))
            if not self._input_speech_active:
                self._input_speech_active = True
                self.emit("input_speech_started", llm.InputSpeechStartedEvent())
            if generation is not None:
                self._close_generation(generation, interrupted=True)
        elif event.type == "canceled":
            if str(data.get("turn_id")) == self._discarded_turn_id:
                self._discarded_turn_id = None
                return
            generation = self._generation_for(data.get("turn_id"))
            if generation is not None:
                generation.generated_audio_ms = 0
                if not self._input_speech_active:
                    self._input_speech_active = True
                    self.emit("input_speech_started", llm.InputSpeechStartedEvent())
                self._close_generation(generation, interrupted=True)
        elif event.type in ("asr", "asr_correction"):
            self._handle_transcription(data, corrected=event.type == "asr_correction")
        elif event.type == "reconnected":
            self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())
        elif event.type in ("error", "resume_failed"):
            detail = str(data.get("detail") or "Dialt realtime error")
            error = llm.RealtimeError(detail, code=cast(str | None, data.get("code")))
            self._emit_error(error, recoverable=bool(data.get("retryable", False)))
            if event.type == "resume_failed":
                self._terminate_session(detail)
        elif event.type == "tool_cancel":
            logger.info("Dialt canceled a tool call", extra={"call_id": data.get("id")})

    def _generation_for(self, turn_id: Any) -> _Generation | None:
        generation = self._current_generation
        if generation is None:
            return None
        return generation if turn_id is None or str(turn_id) == generation.turn_id else None

    def _start_generation(self, turn_id: str) -> None:
        if self._current_generation is not None:
            self._close_generation(self._current_generation, interrupted=True)
        generation = _Generation(
            turn_id=turn_id,
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            text_ch=utils.aio.Chan[str](),
            audio_ch=utils.aio.Chan[rtc.AudioFrame](),
        )
        modalities = asyncio.Future[list[Literal["text", "audio"]]]()
        modalities.set_result(["audio", "text"])
        generation.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=turn_id,
                text_stream=generation.text_ch,
                audio_stream=generation.audio_ch,
                modalities=modalities,
            )
        )
        self._current_generation = generation
        self._recent_generations[turn_id] = generation
        if len(self._recent_generations) > 8:
            self._recent_generations.pop(next(iter(self._recent_generations)))

        pending = next(
            (
                item
                for item in self._pending_replies
                if item.acknowledged and not item.future.done()
            ),
            None,
        )
        event = llm.GenerationCreatedEvent(
            message_stream=generation.message_ch,
            function_stream=generation.function_ch,
            user_initiated=pending is not None,
            response_id=turn_id,
        )
        if pending is not None:
            pending.future.set_result(event)
        self.emit("generation_created", event)

    def _close_generation(self, generation: _Generation, *, interrupted: bool) -> None:
        if generation.done:
            return
        generation.done = True
        generation.text_ch.close()
        generation.audio_ch.close()
        generation.message_ch.close()
        generation.function_ch.close()
        if generation.output_text:
            self._chat_ctx.add_message(
                role="assistant",
                content=generation.output_text,
                id=generation.turn_id,
                interrupted=interrupted,
            )
        if self._current_generation is generation:
            self._current_generation = None

    def _handle_tool_call(self, data: dict[str, Any]) -> None:
        call_id = data.get("id")
        name = data.get("name")
        args = data.get("args")
        if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(args, dict):
            self._emit_error(
                llm.RealtimeError("Dialt sent an invalid tool_call", code="invalid_tool_call"),
                recoverable=True,
            )
            return
        if self._current_generation is None:
            self._start_generation(utils.shortuuid("dialt_tool_"))
        assert self._current_generation is not None
        self._current_generation.function_ch.send_nowait(
            llm.FunctionCall(call_id=call_id, name=name, arguments=json.dumps(args))
        )

    def _handle_transcription(self, data: dict[str, Any], *, corrected: bool) -> None:
        text = data.get("text")
        if not isinstance(text, str) or not text:
            return
        if not self._input_speech_active:
            self.emit("input_speech_started", llm.InputSpeechStartedEvent())
        self._input_speech_active = False
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(user_transcription_enabled=True),
        )
        item_id = str(data.get("message_id") or data.get("turn_id") or utils.shortuuid("dialt_in_"))
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(item_id=item_id, transcript=text, is_final=True),
        )
        existing = self._chat_ctx.get_by_id(item_id)
        if corrected and isinstance(existing, llm.ChatMessage):
            existing.content = [text]
        elif existing is None:
            self._chat_ctx.add_message(role="user", content=text, id=item_id)

    def _emit_error(self, error: Exception, *, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model.label,
                error=error,
                recoverable=recoverable,
            ),
        )
