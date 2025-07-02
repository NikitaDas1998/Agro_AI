# scripts/voice_advisory.py

import os
import requests
import numpy as np
import speech_recognition as sr
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("DUBVERSE_API_KEY")

# Load trained YOLO model
model = YOLO("/Users/nikki/Agro_AI_/runs/classify/train/weights/best.pt")

# Disease → Advisory text mapping
disease_solutions = {
    "Black Rot": {
        "en": "Black Rot detected. Use Mancozeb spray and prune infected leaves.",
        "hi": "ब्लैक रॉट का पता चला है। मेन्कोज़ेब स्प्रे का उपयोग करें और संक्रमित पत्तियों की छंटाई करें।",
        "mr": "काळी कुज आढळली. मॅन्कोझेब स्प्रे वापरा आणि संक्रमित पाने छाटून टाका."
    },
    "Esca": {
        "en": "Esca detected. Remove infected vines and apply proper fungicide.",
        "hi": "एस्का का पता चला। संक्रमित बेलों को हटा दें और उचित कवकनाशी का प्रयोग करें।",
        "mr": "एस्का आढळला. संक्रमित वेली काढून टाका आणि योग्य बुरशीनाशक वापरा."
    },
    "Leaf Blight": {
        "en": "Leaf Blight detected. Apply copper-based fungicides and ensure proper drainage.",
        "hi": "पत्ती झुलसा रोग का पता चला है। कॉपर-आधारित कवकनाशी का प्रयोग करें तथा उचित जल निकासी सुनिश्चित करें।",
        "mr": "पानांवर करपा आढळला. तांबे-आधारित बुरशीनाशके वापरा आणि योग्य निचरा सुनिश्चित करा."
    },
    "Healthy": {
        "en": "The leaf is healthy. No action needed.",
        "hi": "पत्ता स्वस्थ है। कोई कार्रवाई की जरूरत नहीं है।",
        "mr": "पान निरोगी आहे. काही करण्याची गरज नाही."
    }
}

def recognize_speech(prompt_text="🎙️ Speak now"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(prompt_text)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language='hi-IN')
        print("📝 You said:", text)
        return text
    except:
        print("⚠️ Could not recognize speech.")
        return ""

def speak_response(text, lang='en', api_key=None):
    try:
        if api_key:
            print(f"🔊 Using Dubverse TTS for {lang.upper()}...")
            speaker_map = {
                'en': 184,
                'hi': 182,
                'mr': 190

            }
            speaker_id = speaker_map.get(lang, 184)
            dubverse_tts(text, api_key, speaker_no=speaker_id)
        else:
            raise Exception("Dubverse API key not found.")
    except Exception as e:
        print("❌ Speech Error:", e)

def dubverse_tts(text: str, api_key: str, speaker_no: int = 1190, output="response.wav") -> str:
    url = "https://audio.dubverse.ai/api/tts"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "speaker_no": speaker_no,
        "config": {"use_streaming_response": False}
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code == 200:
        with open(output, "wb") as f:
            f.write(resp.content)
        os.system(f"afplay {output}")  # Use 'aplay' or 'ffplay' for Linux
        print(f"✅ Saved audio to {output}")
        return output
    else:
        raise Exception(f"Dubverse API Error {resp.status_code}: {resp.text}")

def ask_language(api_key):
    print("🌐 Which language do you prefer: English, Hindi, or Marathi?")
    speak_response("Which language do you prefer: English, Hindi, or Marathi", 'en', api_key)
    speech = recognize_speech()
    if "hindi" in speech.lower() or "हिंदी" in speech.lower():
        return 'hi'
    elif "marathi" in speech.lower() or "मराठी" in speech.lower():
        return 'mr'
    elif "english" in speech.lower() or "इंग्लिश" in speech.lower():
        return 'en'
    else:
        print("⚠️ Could not detect language. Defaulting to English.")
        return 'en'

def detect_disease(image_path):
    results = model(image_path)
    names = results[0].names
    probs = results[0].probs.data.tolist()
    predicted = names[np.argmax(probs)]
    print("🔍 Predicted Disease:", predicted)
    return predicted

def generate_advisory(disease, lang):
    return disease_solutions.get(disease, {
        'en': "Disease not recognized.",
        'hi': "रोग की पहचान नहीं हुई।",
        'mr': "रोग ओळखता आला नाही."
    })[lang]

def main():
    if not API_KEY:
        print("❌ Dubverse API key not found in .env file.")
        return

    language = ask_language(API_KEY)
    image_path = input("🖼️ Enter path of grape leaf image: ").strip()
    if not os.path.exists(image_path):
        print("❌ Invalid image path.")
        return

    print("📸 Detecting disease...")
    disease = detect_disease(image_path)
    advisory = generate_advisory(disease, language)

    print(f"🗣️ Advisory ({language}):", advisory)
    speak_response(advisory, lang=language, api_key=API_KEY)

if __name__ == "__main__":
    main()