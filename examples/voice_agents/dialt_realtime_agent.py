import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import dialt

load_dotenv()

INSTRUCTIONS = (
    "You are a friendly voice assistant. Keep answers brief and conversational. "
    "Ask one concise clarifying question when a request is missing a key detail."
)

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        llm=dialt.RealtimeModel(
            instructions=INSTRUCTIONS,
            voice=os.getenv("DIALT_VOICE"),
            greeting="Hello, how can I help?",
        )
    )
    await session.start(agent=Agent(instructions=INSTRUCTIONS), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
