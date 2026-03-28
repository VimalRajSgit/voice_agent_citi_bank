from cartesia import Cartesia
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("CARTESIA_API_KEY")
client = Cartesia(api_key=api_key)

print("Starting ffplay...")
player = subprocess.Popen(
    ["ffplay", "-f", "f32le", "-ar", "44100", "-probesize", "32",
     "-analyzeduration", "0", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"],
    stdin=subprocess.PIPE,
    bufsize=0,
)

print("Generating audio...")
audio_chunks = client.tts.generate_sse(   # ✅ updated method name
    model_id="sonic-3",
    transcript="Hi there! Welcome to Cartesia Sonic.",
    voice={
        "mode": "id",
        "id": "a167e0f3-df7e-4d52-a9c3-f949145efdab",
    },
    output_format={
        "container": "raw",
        "encoding": "pcm_f32le",
        "sample_rate": 44100,
    },
    # ✅ removed 'stream=True' — generate_sse() always streams
)

for chunk in audio_chunks:
    if hasattr(chunk, 'audio') and chunk.audio:
        print(f"Received audio chunk ({len(chunk.audio)} bytes)")
        player.stdin.write(chunk.audio)

player.stdin.close()
player.wait()
print("Done!")
