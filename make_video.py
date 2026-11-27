import os
import json
import time
import random
import asyncio
import requests
import yt_dlp
import edge_tts
from PIL import Image
from moviepy import VideoFileClip, ImageClip, AudioFileClip, concatenate_videoclips

GROQ_API_KEY = os.environ["GROQ_API_KEY"].strip()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"].strip()
PIXABAY_API_KEY = os.environ["PIXABAY_API_KEY"].strip()
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"].strip()
VIDEO_URL = os.environ["VIDEO_URL"].strip()

os.makedirs("temp", exist_ok=True)
USED_VIDEO_IDS = set()

def download_audio(url):
    audio_path = "temp/input_audio.mp3"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': "temp/input_audio.%(ext)s",
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '128'}],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return audio_path

def transcribe_audio(audio_path):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    with open(audio_path, "rb") as f:
        res = requests.post(url, headers=headers, files={"file": ("audio.mp3", f, "audio/mpeg")}, data={"model": "whisper-large-v3"})
    data = res.json()
    if "text" not in data:
        raise Exception(f"Groq Error: {data}")
    return data["text"]

def rewrite_script(transcript):
    prompt = f"""
    Convert this transcript into a 30-second viral Hinglish faceless reel script with 6 short punchy scenes (each scene 1 short sentence).
    For each scene, provide 'search_query' (2 simple English stock video keywords matching what is spoken in 'narration', e.g., "stressed man laptop", "money cash", "person running").
    Return ONLY valid JSON:
    {{"scenes": [{{"narration": "Hinglish sentence here", "search_query": "2 english keywords"}}]}}
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
    await edge_tts.Communicate(text, "hi-IN-MadhurNeural", rate="+12%", pitch="+2Hz").save(path)

def download_pixabay_video(query, out_path):
    queries = [query, query.split()[0] if query.split() else "business", "city night"]
    for q in queries:
        try:
            res = requests.get(
                "https://pixabay.com/api/videos/",
                params={"key": PIXABAY_API_KEY, "q": q, "per_page": 10, "safesearch": "true"},
                timeout=20
            ).json()
            hits = [v for v in res.get("hits", []) if v["id"] not in USED_VIDEO_IDS]
            if not hits:
                hits = res.get("hits", [])
            if hits:
                chosen = random.choice(hits[:5])
                USED_VIDEO_IDS.add(chosen["id"])
                videos_dict = chosen.get("videos", {})
                # Prefer medium (720p) or small (540p) MP4 for fast cloud download & crisp quality
                vid_url = (
                    videos_dict.get("medium", {}).get("url")
                    or videos_dict.get("small", {}).get("url")
                    or videos_dict.get("large", {}).get("url")
                )
                if vid_url:
                    vid_data = requests.get(vid_url, timeout=45).content
                    if len(vid_data) > 50000:
                        with open(out_path, "wb") as f:
                            f.write(vid_data)
                        return True
        except Exception:
            pass
    return False

def fit_clip_to_duration(clip, target_duration):
    # Resize and center-crop any horizontal/vertical clip cleanly to 720x1280 (9:16)
    target_w, target_h = 720, 1280
    scale = max(target_w / clip.w, target_h / clip.h)
    clip = clip.resized(scale)
    x_center, y_center = clip.w / 2, clip.h / 2
    clip = clip.cropped(
        x1=max(0, x_center - target_w / 2),
        y1=max(0, y_center - target_h / 2),
        width=target_w,
        height=target_h
    )
    if clip.duration >= target_duration:
        return clip.subclipped(0, target_duration)
    else:
        loops = int(target_duration // clip.duration) + 1
        return concatenate_videoclips([clip] * loops).subclipped(0, target_duration)

async def main():
    audio_file = download_audio(VIDEO_URL)
    transcript = transcribe_audio(audio_file)
    scenes = rewrite_script(transcript)

    clips = []
    for idx, scene in enumerate(scenes):
        v_path = f"temp/v_{idx}.mp3"
        mp4_path = f"temp/clip_{idx}.mp4"
        await generate_voice(scene["narration"], v_path)
        a_clip = AudioFileClip(v_path)

        if download_pixabay_video(scene["search_query"], mp4_path):
            raw_v = VideoFileClip(mp4_path).without_audio()
            v_clip = fit_clip_to_duration(raw_v, a_clip.duration)
        else:
            fallback = Image.new("RGB", (720, 1280), color=(20, 28, 48))
            fallback.save(f"temp/fb_{idx}.jpg")
            v_clip = ImageClip(f"temp/fb_{idx}.jpg").with_duration(a_clip.duration)

        clips.append(v_clip.with_audio(a_clip))

    final = concatenate_videoclips(clips, method="compose")
    out_file = "final_reel.mp4"
    final.write_videofile(out_file, fps=30, codec="libx264", audio_codec="aac", bitrate="2500k", logger=None)

    with open(out_file, "rb") as video:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo",
            data={"chat_id": CHAT_ID, "caption": "✅ Pixabay HD Video Clips + Voiceover Sync Ready!"},
            files={"video": video}
        )

if __name__ == "__main__":
    asyncio.run(main())
