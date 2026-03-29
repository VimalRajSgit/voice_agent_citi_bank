so first i cloned : https://github.com/livekit-examples/voice-pipeline-agent-python/tree/main 
# Linux setup
cd voice-pipeline-agent-python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 agent.py download-files

. so after that i setuped rag system : rag_chat.py 
and then used groq , instead of openai , 
cartesia itself of tts 
groq for stt,llm
