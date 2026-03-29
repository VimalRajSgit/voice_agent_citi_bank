import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    RoomInputOptions,
)
from livekit.plugins import (
    cartesia,
    groq,
    silero,
)

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

class Assistant(Agent):
    def __init__(self, retrieve_context) -> None:
        super().__init__(
            instructions="""You are a CitiBank voice assistant.
Your job is to answer questions using ONLY the context provided to you from official CitiBank documents.
If the context does not contain enough information, say: 'I dont have enough information to answer that.'
Do NOT make up information. Keep answers short and natural sounding like a human customer care assistant.""",
            stt=groq.STT(),
            llm=groq.LLM(model="llama-3.3-70b-versatile"),
            tts=cartesia.TTS(),
        )
        self._retrieve_context = retrieve_context

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the user warmly and ask how you can help them with their CitiBank queries.",
            allow_interruptions=True
        )

    async def on_user_turn_completed(self, turn_ctx, new_message):
        logger.info(f"on_user_turn_completed called!")
        logger.info(f"new_message type: {type(new_message)}")
        logger.info(f"new_message: {new_message}")

        content = new_message.content
        if isinstance(content, list):
            user_question = " ".join([c if isinstance(c, str) else c.get("text", "") for c in content])
        else:
            user_question = content

        logger.info(f"User asked: {user_question}")
        try:
            context = await self._retrieve_context(user_question)
            logger.info(f"RAG context: {context[:200]}")
            if not context.strip():
                logger.warning("RAG returned empty context!")
            turn_ctx.add_message(
                role="system",
                content=f"Use this context from CitiBank documents to answer:\n\n{context}"
            )
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # load RAG during prewarm so it's ready before any job starts
    from rag import retrieve_context
    proc.userdata["retrieve_context"] = retrieve_context
    logger.info("RAG system preloaded successfully")

async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    retrieve_context = ctx.proc.userdata["retrieve_context"]

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0,
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(retrieve_context=retrieve_context),
        room_input_options=RoomInputOptions(
            noise_cancellation=None,
        ),
    )

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            initialize_process_timeout=60.0,  # give 60 seconds to load
        ),
    )
