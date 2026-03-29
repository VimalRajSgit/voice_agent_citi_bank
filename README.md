# Setup

so First cloned the repo:
https://github.com/livekit-examples/voice-pipeline-agent-python

```bash
cd voice-pipeline-agent-python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 agent.py download-files
```

After setup, added a RAG system (`rag_chat.py`).

## Changes

* Replaced OpenAI with Groq
* Used Groq for STT + LLM
* Used Cartesia for TTS


## Screen shot
![screenshot](./voice_agent.png)
 
#Performance
latency is around 1000ms-1200ms , the default agent code had around 500ms , so rag system had to run within 500-520MS , for that i decreased the retriver function from 10 to 3 n i used a smaller llama instant model instead, 
and there is no seperate llm for rag system , 
* User speaks → Groq STT converts to text
* Text query → Pinecone finds similar vectors → returns raw text chunks
* Raw text chunks injected into agent context
* Groq LLM reads context + answers
* Cartesia TTS speaks the answer
