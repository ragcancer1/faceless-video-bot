import os
import io
import json
import math
import time
import random
import asyncio
import requests
import yt_dlp
import edge_tts
from gtts import gTTS
from PIL import Image, ImageStat
from moviepy import VideoFileClip, ImageClip, AudioFileClip, concatenate_videoclips

GROQ_API_KEY = os.environ["GROQ_API_KEY"].strip()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"].strip()
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY", "").strip()
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"].strip()
VIDEO_URL = os.environ["VIDEO_URL"].strip()

os.makedirs("temp", exist_ok=True)
USED_VIDEO_IDS = set()

STYLE_SUFFIX = "3D Pixar Disney animation style, vibrant cinematic lighting, 9:16 vertical, 8k, no text, no watermark"

def download_audio_and_get_duration(url):
    audio_path = "temp/input_audio.mp3"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': "temp/input_audio.%(ext)s",
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '128'}],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        duration = float(info.get("duration") or 30.0)
    duration = max(12.0, min(60.0, duration))
    return audio_path, duration

def transcribe_audio(audio_path):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    with open(audio_path, "rb") as f:
        res = requests.post(url, headers=headers, files={"file": ("audio.mp3", f, "audio/mpeg")}, data={"model": "whisper-large-v3"})
    data = res.json()
    if "text" not in data:
        raise Exception(f"Groq Error: {data}")
    return data["text"]

def rewrite_script(transcript, target_duration):
    num_scenes = max(4, min(10, round(target_duration / 4.5)))
    words_per_scene = max(7, round((target_duration * 2.3) / num_scenes))

    prompt = f"""
    Rewrite this transcript into an engaging viral Hindi animated story reel of EXACTLY {round(target_duration)} seconds divided into {num_scenes} scenes.
    
    CRITICAL RULES:
    1. 'voice_hindi': Pure natural Devanagari Hindi script with Hindi words for ALL numbers (e.g., "दस अवतार", "बीस साल", "सात हज़ार"). Keep each scene around {words_per_scene} Hindi words so total narration matches ~{round(target_duration)} seconds.
    2. 'visual_prompt': Short, vivid English prompt (MAX 22 words) describing the 3D animated scene, main character, action, and background. Keep character style consistent across scenes.
    3. 'fallback_keyword': 2 simple English words for stock video backup if AI render times out (e.g., "ocean waves", "temple light").

    Return ONLY valid JSON:
    {{"scenes": [{{"voice_hindi": "...", "visual_prompt": "...", "fallback_keyword": "..."}}]}}
    
    Transcript: {transcript}
    """
    gemini_res = {}
    for model in ["gemini-3.8-flash", "gemini-2.5-flash-lite"]:
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        for _ in range(2):
            gemini_res = requests.post(gemini_url, json={"contents": [{"parts": [{"text": prompt}]}]}).json()
            if "candidates" in gemini_res:
                raw = gemini_res["candidates"][0]["content"]["parts"][0]["text"].replace("```json", "").replace("```", "").strip()
                return json.loads(raw)["scenes"]
            time.sleep(3)

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    models_data = requests.get("https://api.groq.com/openai/v1/models", headers=headers).json()
    available_models = [
        m["id"] for m in models_data.get("data", [])
        if any(k in m["id"].lower() for k in ["llama", "qwen", "gemma", "deepseek"])
        and "whisper" not in m["id"].lower() and "guard" not in m["id"].lower()
    ]
    
    groq_res = {}
    for g_model in available_models:
        groq_res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": g_model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            }
        ).json()
        if "choices" in groq_res:
            raw = groq_res["choices"][0]["message"]["content"].strip()
            return json.loads(raw)["scenes"]

    raise Exception(f"Both APIs failed. Gemini: {gemini_res} | Groq: {groq_res}")

async def generate_voice(text, path):
    clean_text = text.strip() or "कहानी आगे बढ़ती है।"
    for voice in ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"]:
        for attempt in range(3):
            try:
                await edge_tts.Communicate(clean_text, voice, rate="+6%").save(path)
                if os.path.exists(path) and os.path.getsize(path) > 1000:
                    await asyncio.sleep(1)
                    return
            except Exception:
                await asyncio.sleep(2)
    gTTS(text=clean_text, lang="hi", slow=False).save(path)

def is_valid_artwork(img):
    stat = ImageStat.Stat(img)
    mean_brightness = sum(stat.mean) / 3
    std_dev = sum(stat.stddev) / 3
    return (20 < mean_brightness < 240) and (std_dev > 18)

def create_motion_clip_from_image(img_path, target_dur, idx):
    base = ImageClip(img_path).with_duration(target_dur)
    if idx % 3 == 0:
        return base.resized(lambda t, d=target_dur: 1.0 + 0.10 * (t / d)).with_position(('center', 'center'))
    elif idx % 3 == 1:
        return base.resized(lambda t, d=target_dur: 1.10 - 0.10 * (t / d)).with_position(('center', 'center'))
    else:
        return base.resized(1.12).with_position(lambda t, d=target_dur: (int(-35 * (t / d)), 'center'))

def render_scene_visual(visual_prompt, fallback_kw, out_mp4, target_dur, idx):
    short_prompt = f"{visual_prompt[:180]}, {STYLE_SUFFIX}"
    encoded = requests.utils.quote(short_prompt)

    endpoints = [
        f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1360&model=flux&nologo=true&seed={400 + idx * 23}",
        f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1360&model=turbo&nologo=true&seed={500 + idx * 23}",
        f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1360&nologo=true&seed={600 + idx * 23}"
    ]

    for attempt in range(9):
        url = endpoints[attempt % len(endpoints)]
        try:
            res = requests.get(url, timeout=75)
            if res.status_code == 200 and len(res.content) > 25000:
                candidate = Image.open(io.BytesIO(res.content)).convert("RGB")
                if is_valid_artwork(candidate):
                    w, h = candidate.size
                    img = candidate.crop((0, 0, w, int(h * 0.93))).resize((720, 1280), Image.Resampling.LANCZOS)
                    img_path = f"temp/i_{idx}.jpg"
                    img.save(img_path, "JPEG", quality=95)
                    ic = create_motion_clip_from_image(img_path, target_dur, idx)
                    ic.write_videofile(out_mp4, fps=30, codec="libx264", audio=False, logger=None)
                    print(f"Scene {idx+1}: 3D Animated frame rendered!")
                    time.sleep(5)
                    return
        except Exception as e:
            print(f"Scene {idx+1} attempt {attempt+1} retrying: {e}")
        time.sleep(6)

    if PIXABAY_API_KEY:
        try:
            res = requests.get(
                "https://pixabay.com/api/videos/",
                params={"key": PIXABAY_API_KEY, "q": fallback_kw or "universe galaxy", "per_page": 6},
                timeout=20
            ).json()
            hits = [v for v in res.get("hits", []) if v["id"] not in USED_VIDEO_IDS] or res.get("hits", [])
            if hits:
                chosen = random.choice(hits)
                USED_VIDEO_IDS.add(chosen["id"])
                vid_url = chosen.get("videos", {}).get("medium", {}).get("url")
                if vid_url:
                    raw_p = f"temp/pb_{idx}.
