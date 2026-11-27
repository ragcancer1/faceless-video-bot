import os
import io
import json
import time
import asyncio
import requests
import yt_dlp
import edge_tts
from PIL import Image, ImageStat
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

GROQ_API_KEY = os.environ["GROQ_API_KEY"].strip()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"].strip()
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"].strip()
VIDEO_URL = os.environ["VIDEO_URL"].strip()

os.makedirs("temp", exist_ok=True)

STYLE_SUFFIX = (
    "3D Pixar Disney animated movie scene, expressive characters, warm cinematic lighting, "
    "rich detailed full-bleed background, 9:16 vertical composition, Unreal Engine 5 render, "
    "vibrant colors, sharp focus, no text, no subtitles, no watermark"
)

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
    Rewrite this story/transcript into a gripping 35-second viral Hindi animated story reel with 6 scenes (each scene 1 short punchy sentence).
    
    CRITICAL RULES:
    1. 'voice_hindi': Write in pure natural Devanagari Hindi script with Hindi words for ALL numbers (e.g., "बीस साल", "सात हज़ार", "दो रेत के बोरे") so text-to-speech sounds completely human and natural.
    2. 'character_sheet': A single 1-sentence English description of the main character(s) and setting (e.g., "a 45-year-old Indian customs officer with a mustache in a khaki uniform and a young smuggler in a brown jacket with a bicycle carrying two sandbags at a sunlit border gate").
    3. 'scene_action': Specific English visual action and camera shot (wide shot, medium shot, close-up) for this exact scene, repeating the exact characters from 'character_sheet' so every frame looks like the same animated movie.

    Return ONLY valid JSON:
    {{"character_sheet": "...", "scenes": [{{"voice_hindi": "...", "scene_action": "..."}}]}}
    
    Transcript: {transcript}
    """
    gemini_res = {}
    for model in ["gemini-3.8-flash", "gemini-2.5-flash-lite"]:
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        for _ in range(2):
            gemini_res = requests.post(gemini_url, json={"contents": [{"parts": [{"text": prompt}]}]}).json()
            if "candidates" in gemini_res:
                raw = gemini_res["candidates"][0]["content"]["parts"][0]["text"].replace("```json", "").replace("```", "").strip()
                return json.loads(raw)
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
            return json.loads(raw)

    raise Exception(f"Both APIs failed. Gemini: {gemini_res} | Groq: {groq_res}")

async def generate_voice(text, path):
    await edge_tts.Communicate(text, "hi-IN-MadhurNeural", rate="+6%", pitch="+0Hz").save(path)

def is_valid_artwork(img):
    stat = ImageStat.Stat(img)
    mean_brightness = sum(stat.mean) / 3
    std_dev = sum(stat.stddev) / 3
    # Reject solid blank frames, pure black, or pure white images
    return (20 < mean_brightness < 240) and (std_dev > 18)

def generate_animated_frame(character_sheet, scene_action, out_path, idx):
    full_prompt = f"{scene_action}, featuring {character_sheet}, {STYLE_SUFFIX}"
    encoded = requests.utils.quote(full_prompt)

    endpoints = [
        f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1360&model=flux&nologo=true&enhance=true&seed={500 + idx * 19}",
        f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1360&model=turbo&nologo=true&seed={600 + idx * 19}",
        f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1360&nologo=true&seed={700 + idx * 19}",
    ]

    # Patiently retry up to 9 times across endpoints so zero blank frames ever happen
    for attempt in range(9):
        url = endpoints[attempt % len(endpoints)]
        try:
            res = requests.get(url, timeout=90)
            if res.status_code == 200 and len(res.content) > 30000:
                candidate = Image.open(io.BytesIO(res.content)).convert("RGB")
                if is_valid_artwork(candidate):
                    w, h = candidate.size
                    # Crop bottom 7% to remove any watermark cleanly
                    img = candidate.crop((0, 0, w, int(h * 0.93))).resize((720, 1280), Image.Resampling.LANCZOS)
                    img.save(out_path, "JPEG", quality=95)
                    print(f"Scene {idx+1} rendered cleanly on attempt {attempt+1}")
                    time.sleep(8)  # Cooldown to avoid rate limits on next scene
                    return
        except Exception as e:
            print(f"Scene {idx+1} attempt {attempt+1} retrying: {e}")
        time.sleep(10)

    raise Exception(f"Failed to generate valid 3D animation frame for scene {idx+1} after 9 attempts.")

async def main():
    audio_file = download_audio(VIDEO_URL)
    transcript = transcribe_audio(audio_file)
    story_data = rewrite_script(transcript)
    character_sheet = story_data.get("character_sheet", "")
    scenes = story_data["scenes"]

    clips = []
    for idx, scene in enumerate(scenes):
        v_path = f"temp/v_{idx}.mp3"
        i_path = f"temp/i_{idx}.jpg"
        await generate_voice(scene["voice_hindi"], v_path)
        generate_animated_frame(character_sheet, scene["scene_action"], i_path, idx)

        a_clip = AudioFileClip(v_path)
        dur = max(a_clip.duration, 1.5)

        # Smooth alternating camera motion (zoom-in / zoom-out)
        if idx % 2 == 0:
            i_clip = (
                ImageClip(i_path)
                .with_duration(dur)
                .resized(lambda t, d=dur: 1.0 + 0.08 * (t / d))
                .with_position(('center', 'center'))
            )
        else:
            i_clip = (
                ImageClip(i_path)
                .with_duration(dur)
                .resized(lambda t, d=dur: 1.08 - 0.08 * (t / d))
                .with_position(('center', 'center'))
            )

        clips.append(i_clip.with_audio(a_clip))

    final = concatenate_videoclips(clips, method="compose")
    out_file = "final_reel.mp4"
    final.write_videofile(out_file, fps=30, codec="libx264", audio_codec="aac", bitrate="3000k", logger=None)

    with open(out_file, "rb") as video:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo",
            data={"chat_id": CHAT_ID, "caption": "✅ Clean 3D Animated Story Reel Ready (No Captions, Zero Blank Frames)!"},
            files={"video": video}
        )

if __name__ == "__main__":
    asyncio.run(main())
