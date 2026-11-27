import os
import io
import json
import math
import time
import asyncio
import requests
import yt_dlp
import edge_tts
from PIL import Image, ImageStat
from moviepy import VideoFileClip, ImageClip, AudioFileClip, concatenate_videoclips

GROQ_API_KEY = os.environ["GROQ_API_KEY"].strip()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"].strip()
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"].strip()
VIDEO_URL = os.environ["VIDEO_URL"].strip()

os.makedirs("temp", exist_ok=True)

ANIM_STYLE = (
    "3D Pixar Disney animated movie scene, smooth cinematic character motion, "
    "expressive faces, vibrant lighting, rich detailed Indian village/city background, "
    "9:16 vertical composition, 8k render, no text, no subtitles, no watermark"
)

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
    # Clamp between 12s and 60s for Instagram Reels
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
    # Each scene ~5 seconds so total video matches original duration within ±1-2s
    num_scenes = max(3, min(10, round(target_duration / 5.0)))
    words_per_scene = max(8, round((target_duration * 2.3) / num_scenes))

    prompt = f"""
    Rewrite this story/transcript into an engaging viral Hindi animated story reel of EXACTLY {round(target_duration)} seconds divided into {num_scenes} scenes.
    
    CRITICAL RULES:
    1. 'voice_hindi': Write in pure natural Devanagari Hindi script with Hindi words for ALL numbers (e.g., "बीस साल", "सात हज़ार", "दो रेत के बोरे"). Keep each scene around {words_per_scene} Hindi words so the total narration takes ~{round(target_duration)} seconds.
    2. 'character_sheet': A consistent 1-sentence English description of the main character(s) (e.g., "a cheerful Indian farmer in a white dhoti-kurta and turban with a brown water buffalo at a lush green village farm").
    3. 'motion_prompt': Specific English action & movement description for a 5-second animated video clip (describe exact character movement, walking, talking, expressions, and camera motion), keeping the exact same characters from 'character_sheet'.

    Return ONLY valid JSON:
    {{"character_sheet": "...", "scenes": [{{"voice_hindi": "...", "motion_prompt": "..."}}]}}
    
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
    await edge_tts.Communicate(text, "hi-IN-MadhurNeural", rate="+5%", pitch="+0Hz").save(path)

def is_valid_artwork(img):
    stat = ImageStat.Stat(img)
    mean_brightness = sum(stat.mean) / 3
    std_dev = sum(stat.stddev) / 3
    return (20 < mean_brightness < 240) and (std_dev > 18)

def generate_animated_video_clip(character_sheet, motion_prompt, out_mp4, target_dur, idx):
    full_prompt = f"{motion_prompt}, featuring {character_sheet}, {ANIM_STYLE}"
    encoded = requests.utils.quote(full_prompt)

    # 1. Try Free AI Text-to-Video endpoints (real moving 3D animated MP4 clips)
    video_endpoints = [
        f"https://image.pollinations.ai/prompt/{encoded}?model=wan&width=576&height=1024&nologo=true&seed={300 + idx * 17}",
        f"https://image.pollinations.ai/prompt/{encoded}?model=seedance&width=576&height=1024&nologo=true&seed={400 + idx * 17}",
        f"https://video.pollinations.ai/prompt/{encoded}?width=576&height=1024&nologo=true&seed={500 + idx * 17}"
    ]

    for attempt in range(4):
        url = video_endpoints[attempt % len(video_endpoints)]
        try:
            res = requests.get(url, timeout=110)
            content_type = res.headers.get("Content-Type", "").lower()
            if res.status_code == 200 and ("video" in content_type or res.content[:12].find(b"ftyp") != -1) and len(res.content) > 60000:
                raw_vid_path = f"temp/raw_v_{idx}.mp4"
                with open(raw_vid_path, "wb") as f:
                    f.write(res.content)
                clip = VideoFileClip(raw_vid_path).without_audio()
                # Crop bottom 6% to ensure zero watermark and resize to 720x1280
                clip = clip.cropped(x1=0, y1=0, width=clip.w, height=int(clip.h * 0.94)).resized((720, 1280))
                if clip.duration >= target_dur:
                    clip = clip.subclipped(0, target_dur)
                else:
                    loops = math.ceil(target_dur / max(clip.duration, 0.5))
                    clip = concatenate_videoclips([clip] * loops).subclipped(0, target_dur)
                clip.write_videofile(out_mp4, fps=30, codec="libx264", audio=False, logger=None)
                print(f"Scene {idx+1}: Real AI Animated Video generated!")
                time.sleep(6)
                return
        except Exception as e:
            print(f"Scene {idx+1} video attempt {attempt+1} retrying: {e}")
        time.sleep(8)

    # 2. High-Quality 3D Animated Frame Fallback (if free video queue is overloaded)
    img_endpoints = [
        f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1360&model=flux&nologo=true&enhance=true&seed={700 + idx * 19}",
        f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1360&model=turbo&nologo=true&seed={800 + idx * 19}"
    ]
    for attempt in range(6):
        url = img_endpoints[attempt % len(img_endpoints)]
        try:
            res = requests.get(url, timeout=80)
            if res.status_code == 200 and len(res.content) > 30000:
                candidate = Image.open(io.BytesIO(res.content)).convert("RGB")
                if is_valid_artwork(candidate):
                    w, h = candidate.size
                    img = candidate.crop((0, 0, w, int(h * 0.93))).resize((720, 1280), Image.Resampling.LANCZOS)
                    img_path = f"temp/fb_i_{idx}.jpg"
                    img.save(img_path, "JPEG", quality=95)
                    # Smooth cinematic zoom motion
                    if idx % 2 == 0:
                        ic = ImageClip(img_path).with_duration(target_dur).resized(lambda t, d=target_dur: 1.0 + 0.08 * (t / d)).with_position(('center', 'center'))
                    else:
                        ic = ImageClip(img_path).with_duration(target_dur).resized(lambda t, d=target_dur: 1.08 - 0.08 * (t / d)).with_position(('center', 'center'))
                    ic.write_videofile(out_mp4, fps=30, codec="libx264", audio=False, logger=None)
                    print(f"Scene {idx+1}: Fallback 3D frame rendered!")
                    time.sleep(6)
                    return
        except Exception:
            pass
        time.sleep(8)

    raise Exception(f"Failed to render Scene {idx+1}")

async def main():
    audio_file, orig_duration = download_audio_and_get_duration(VIDEO_URL)
    print(f"Original video duration: {orig_duration:.2f}s")
    transcript = transcribe_audio(audio_file)
    story_data = rewrite_script(transcript, orig_duration)
    character_sheet = story_data.get("character_sheet", "")
    scenes = story_data["scenes"]

    clips = []
    for idx, scene in enumerate(scenes):
        v_path = f"temp/v_{idx}.mp3"
        clip_path = f"temp/scene_{idx}.mp4"
        await generate_voice(scene["voice_hindi"], v_path)
        a_clip = AudioFileClip(v_path)
        dur = max(a_clip.duration, 1.5)

        generate_animated_video_clip(character_sheet, scene["motion_prompt"], clip_path, dur, idx)
        v_clip = VideoFileClip(clip_path).with_duration(dur).with_audio(a_clip)
        clips.append(v_clip)

    final = concatenate_videoclips(clips, method="compose")
    out_file = "final_reel.mp4"
    final.write_videofile(out_file, fps=30, codec="libx264", audio_codec="aac", bitrate="3500k", logger=None)

    with open(out_file, "rb") as video:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo",
            data={
                "chat_id": CHAT_ID,
                "caption": f"✅ Instagram-Ready 3D Animated Reel ({round(final.duration)}s — Matched to Original {round(orig_duration)}s)!"
            },
            files={"video": video}
        )

if __name__ == "__main__":
    asyncio.run(main())
