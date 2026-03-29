
import asyncio
from groq import Groq
from livekit import api
import os
import openai
from dotenv import load_dotenv
load_dotenv()
groq_api = os.environ.get("GROQ_API_KEY")
LIVEKIT_URL = os.environ.get("LIVEKIT_URL")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET")


async def text_to_speech(text: str, output_file: str = "output.mp3"):
    """
    Convert text to speech using Groq and LiveKit
    """
    try:
        # Initialize Groq client
        groq_client = Groq(api_key=groq_api)

        # Get TTS from Groq (using OpenAI-compatible API)
        response = groq_client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
            input=text,
            response_format="mp3"
        )

        # Save audio to file
        response.stream_to_file(output_file)
        print(f"Audio saved to {output_file}")

        # Optional: Stream to LiveKit room
        await stream_to_livekit(output_file)

        return output_file

    except Exception as e:
        print(f"Error: {e}")
        return None

async def stream_to_livekit(audio_file: str):
    """
    Stream audio to a LiveKit room
    """
    try:
        # Initialize LiveKit client
        livekit = api.LiveKitAPI(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        )

        # Create or join a room
        room_name = "tts-room"

        # Get room token
        token = livekit.access_token(
            room_name=room_name,
            participant_name="tts-bot",
            participant_identity="tts-bot-1",
            grants=api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True
            )
        )

        print(f"LiveKit token generated. Room: {room_name}")
        print(f"Token: {token}")

        # Note: For actual audio streaming, you'd need the LiveKit SDK
        # This is a simplified example showing token generation

    except Exception as e:
        print(f"LiveKit error: {e}")

# Run the TTS
async def main():
    text = "Hello, this is a test of text to speech using Groq and LiveKit."

    # Generate TTS audio
    audio_file = await text_to_speech(text, "greeting.mp3")

    if audio_file:
        print(f"TTS completed successfully. File: {audio_file}")

if __name__ == "__main__":
    asyncio.run(main())
