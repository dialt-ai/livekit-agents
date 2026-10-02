# LiveKit Dialt Plugin

LiveKit Agents realtime-model integration for Dialt. Dialt owns speech recognition,
turn detection, conversation state, tool orchestration, speech generation, and session resume;
the plugin translates media and events to LiveKit's realtime APIs.

## Installation

```bash
pip install livekit-plugins-dialt
export DIALT_API_KEY="your_api_key"
```

`dialt-sdk==0.41.0` is installed with the plugin.

## Usage

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import dialt

instructions = (
    "You are a concise voice assistant helping callers with their account. "
    "Ask one clear question when required information is missing."
)

session = AgentSession(
    llm=dialt.RealtimeModel(
        instructions=instructions,
        voice="chime",
        greeting="Hello, how can I help?",
    )
)

await session.start(
    agent=Agent(instructions=instructions),
    room=ctx.room,
)
```

Instructions are required because Dialt's platform prompt does not define the agent's role.
The optional `greeting` is a normal, interruptible Dialt turn; do not generate a second greeting
from `Agent.on_enter` when it is configured.

## Tools

LiveKit function tools are sent to Dialt and remain mutable during a session. Additional Dialt
tool behavior is configured by registered tool name:

```python
llm=dialt.RealtimeModel(
    instructions=instructions,
    tool_options={
        "lookup_order": dialt.ToolOptions(read_only=True, expected_duration="instant"),
        "issue_refund": dialt.ToolOptions(
            requires_permission=True,
            permission_source="caller",
            timeout=30,
        ),
    },
)
```

Supported options are `read_only`, `requires_permission`, `permission_source`, `resolver_only`,
`expected_duration`, `status_label`, `timeout`, `deferred`, `deferred_timeout`, and
`notify_on_complete`.

The plugin uses Dialt SDK automatic reconnect and resume by default. Resume tokens remain inside
the SDK and are not exposed by the plugin. Set `auto_reconnect=False` only when the application
must treat every transport loss as terminal.
