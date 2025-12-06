from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "lEvO API is running!"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_prompt = data.get("prompt", "")

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are lEvO — a fast, funny, confident Gen-Z AI created for LeCore. "
                    "You speak casually, stay helpful, and always keep responses accurate, "
                    "short, and energetic. Do NOT speak like an old robot."
                )
            },
            {"role": "user", "content": user_prompt}
        ]
    )

    reply = completion.choices[0].message.content
    return {"response": reply}


