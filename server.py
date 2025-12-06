from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from groq import Groq
import os

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")

        if prompt == "":
            return JSONResponse({"response": "No prompt provided"}, status_code=400)

        completion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are lEvO, a friendly intelligent AI."},
                {"role": "user", "content": prompt},
            ]
        )

        reply = completion.choices[0].message.content  # CORRECT

        return {"response": reply}

    except Exception as e:
        print("🔥 ERROR:", e)          # <--- PRINT REAL ERROR
        return JSONResponse({"response": f"Server crashed: {str(e)}"}, status_code=500)

@app.get("/")
def home():
    return {"status": "lEvO API running!"}
