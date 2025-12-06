import os
import sqlite3
import base64
import requests
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydub import AudioSegment
from groq import Groq

# ----------------------
#  CONFIG + CLIENT
# ----------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
You are lEvO — a hype, super-fast, Gen-Z AI.
You reply short, confident, funny, helpful and energetic.
Don’t act robotic. Don’t be boring. Keep the vibe real.
"""


# ----------------------
#  MEMORY (SQLite)
# ----------------------
conn = sqlite3.connect("memory.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS memory (user_id TEXT, text TEXT)")
conn.commit()

def get_memory(uid):
    cursor.execute("SELECT text FROM memory WHERE user_id=?", (uid,))
    return "\n".join([row[0] for row in cursor.fetchall()])

def save_memory(uid, msg):
    cursor.execute("INSERT INTO memory VALUES (?, ?)", (uid, msg))
    conn.commit()


# ----------------------
#  HOME
# ----------------------
@app.get("/")
def home():
    return {"status": "lEvO API is live 🔥"}


# ----------------------
#  NORMAL CHAT
# ----------------------
@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    user_id = data.get("user_id", "guest")

    memory = get_memory(user_id)

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"User memory:\n{memory}"},
        {"role": "user", "content": prompt}
    ]

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=msgs
        )

        reply = completion.choices[0].message.content

        save_memory(user_id, prompt)
        save_memory(user_id, reply)

        return {"response": reply}

    except Exception as e:
        return {"error": str(e)}


# ----------------------
#  STREAMING CHAT (typing effect)
# ----------------------
@app.get("/stream")
async def stream(prompt: str):
    def generate():
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield f"data: {chunk.choices[0].delta.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ----------------------
#  IMAGE UNDERSTANDING
# ----------------------
@app.post("/vision")
async def vision(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_b64 = base64.b64encode(img_bytes).decode()

    try:
        completion = client.chat.completions.create(
            model="llava-v1.6",
            messages=[
                {"role": "user", "content": [
                    {"type": "input_text", "text": "Describe this image clearly"},
                    {"type": "input_image", "image": img_b64},
                ]}
            ]
        )

        return {"response": completion.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}


# ----------------------
#  GOOGLE SEARCH / WIKIPEDIA RAG
# ----------------------
@app.get("/search")
def search(q: str):
    url = f"https://ddg-api.herokuapp.com/search?query={q}"
    try:
        r = requests.get(url).json()
        return r
    except:
        return {"error": "Search failed"}


@app.post("/chat-rag")
async def chat_rag(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    search_results = search(prompt)

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Search data:\n{search_results}"},
            {"role": "user", "content": prompt}
        ]
    )

    return {"response": completion.choices[0].message.content}


# ----------------------
#  TYPING INDICATOR
# ----------------------
@app.get("/typing")
def typing():
    return {"typing": True}


# ----------------------
#  VOICE → TEXT
# ----------------------
@app.post("/speech")
async def speech(file: UploadFile = File(...)):
    audio = await file.read()

    try:
        result = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=("voice.wav", audio)
        )
        return {"text": result.text}

    except Exception as e:
        return {"error": str(e)}


# ----------------------
#  TEXT → SPEECH  (mp3 base64)
# ----------------------
@app.post("/tts")
async def tts(request: Request):
    data = await request.json()
    text = data.get("text", "")

    resp = client.audio.speech.create(
        model="gpt-4o-mini-tts",  # free + fast
        voice="alloy",
        input=text
    )

    audio_b64 = base64.b64encode(resp.read()).decode()

    return {"audio": audio_b64}
