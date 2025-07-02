from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import os
import shutil
import uuid
import traceback

from scripts.voice_advisory import detect_disease, generate_advisory, speak_response

app = FastAPI()

@app.post("/analyze/")
async def analyze(image: UploadFile = File(...), lang: str = Form(...)):
    try:
        # Save image temporarily
        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        disease = detect_disease(temp_filename)
        advisory = generate_advisory(disease, lang)
        speak_response(advisory, lang=lang, api_key=os.getenv("DUBVERSE_API_KEY"))

        os.remove(temp_filename)
        return {"disease": disease, "advisory": advisory}

    except Exception as e:
        print("❌ Internal Server Error:", e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
