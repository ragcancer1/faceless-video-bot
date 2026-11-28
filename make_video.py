import os
import io
import json
import time
import random
import asyncio
import subprocess
import requests
import yt_dlp
import edge_tts
from gtts import gTTS
from PIL import Image, ImageDraw, ImageStat

GROQ_API_KEY = os.environ["GROQ_API_KEY"].strip()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"].strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"].strip()
VIDEO_URL = os.environ["VIDEO_URL"].strip()

os.makedirs("temp", exist_ok=True)

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
    return audio_path, max(12.0, min(60.0, duration))

def transcribe_audio(audio_path):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    with open(audio_path, "rb") as f:
        res = requests.post(url, headers=headers, files={"file": ("audio.mp3", f, "audio/mpeg")}, data={"model": "whisper-large-v3"})
    data = res.json()
    if "text" not in data:
        raise Exception(f"Groq Whisper Error: {data}")
    return data["text"]

def rewrite_script(transcript, target_duration):
    num_scenes = max(4, min(7, round(target_duration / 5.5)))
    words_per_scene = max(8, round((target_duration * 2.2) / num_scenes))

    prompt = f"""
    Rewrite this viral story transcript into a {round(target_duration)}-second Hindi 3D animated reel with {num_scenes} scenes.

    STRICT RULES:
    1. 'character_bible': Define ONE consistent 3D Pixar-style main character and setting in English (MAX 15 words).
    2. 'voice_hindi': Pure Devanagari Hindi narration matching the story's emotion. Write ALL numbers in Hindi words (e.g., "दस", "बीस साल"). Keep each scene ~{words_per_scene} Hindi words.
    3. 'visual_prompt': Specific scene action (MAX 18 words) using the EXACT same character from 'character_bible'.

    Return ONLY valid JSON:
    {{"character_bible": "...", "scenes": [{{"voice_hindi": "...", "visual_prompt": "..."}}]}}

    Transcript: {transcript}
    """
    gemini_res = {}
    for model in ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]:
        g_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        for _ in range(2):
            gemini_res = requests.post(g_url, json={"contents": [{"parts": [{"text": prompt}]}]}).json()
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
    clean_text = text.strip() or "कहानी आगे बढ़ती है।"
    for voice in ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"]:
        for _ in range(3):
            try:
                await edge_tts.Communicate(clean_text, voice, rate="+5%").save(path)
                if os.path.exists(path) and os.path.getsize(path) > 1000:
                    return
            except Exception:
                await asyncio.sleep(1.5)
    gTTS(text=clean_text, lang="hi", slow=False).save(path)

def get_audio_duration(audio_path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
    return max(1.5, float(subprocess.check_output(cmd).decode().strip()))

def is_clean_image(img):
    stat = ImageStat.Stat(img)
    return (20 < sum(stat.mean) / 3 < 240) and (sum(stat.stddev) / 3 > 20)

def generate_consistent_3d_image(character_bible, visual_prompt, out_jpg, idx):
    full_prompt = (
        f"3D Pixar Disney animated movie scene, {visual_prompt}, featuring {character_bible}, "
        f"warm cinematic lighting, vertical 9:16 framing, no text, no watermark"
    )

    # 1. Hugging Face Free Inference API (Primary — Fast & Consistent if HF_TOKEN is set)
    if HF_TOKEN:
        hf_models = [
            "black-forest-labs/FLUX.1-schnell",
            "stabilityai/stable-diffusion-xl-base-1.0"
        ]
        hf_headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        for hf_m in hf_models:
            try:
                res = requests.post(
                    f"https://router.huggingface.co/hf-inference/models/{hf_m}",
                    headers=hf_headers,
                    json={"inputs": full_prompt, "parameters": {"width": 720, "height": 1280}},
                    timeout=60
                )
                if res.status_code == 200 and len(res.content) > 25000:
                    cand = Image.open(io.BytesIO(res.content)).convert("RGB")
                    if is_clean_image(cand):
                        cand.resize((720, 1280), Image.Resampling.LANCZOS).save(out_jpg, "JPEG", quality=95)
                        print(f"Scene {idx+1}: HuggingFace {hf_m} rendered!")
                        return
            except Exception as e:
                print(f"HF {hf_m} skipped: {e}")

    # 2. Pollinations with 15s Cooldown (Respects free rate limit so Scene 2+ never block)
    encoded = requests.utils.quote(full_prompt[:200])
    for attempt in range(6):
        model_name = "flux" if attempt % 2 == 0 else "turbo"
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1360&model={model_name}&nologo=true&seed={random.randint(1000, 99999)}"
        try:
            res = requests.get(url, timeout=60)
            if res.status_code == 200 and len(res.content) > 30000:
                cand = Image.open(io.BytesIO(res.content)).convert("RGB")
                if is_clean_image(cand):
                    w, h = cand.size
                    cand.crop((0, 0, w, int(h * 0.93))).resize((720, 1280), Image.Resampling.LANCZOS).save(out_jpg, "JPEG", quality=95)
                    print(f"Scene {idx+1}: Pollinations rendered!")
                    time.sleep(15)
                    return
        except Exception:
            pass
        time.sleep(15)

    # 3. Last-resort smooth gradient card (never crashes workflow)
    img = Image.new("RGB", (720, 1280), color=(25, 35, 65))
    draw = ImageDraw.Draw(img)
    for y in range(1280):
        draw.line([(0, y), (720, y)], fill=(20 + int(35 * y / 1280), 30 + int(45 * y / 1280), 65 + int(75 * y / 1280)))
    img.save(out_jpg, "JPEG", quality=90)
    print(f"Scene {idx+1}: Styled fallback card used.")

def render_scene_with_ffmpeg(img_path, audio_path, out_mp4, dur, idx):
    frames = int(dur * 30)
    if idx % 2 == 0:
        zp = f"zoompan=z='min(zoom+0.0012,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=720x1280:fps=30"
    else:
        zp = f"zoompan=z='if(lte(zoom,1.0),1.12,max(1.001,zoom-0.0012))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=720x1280:fps=30"

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", img_path,
        "-i", audio_path,
        "-vf", f"scale=1440:2560,{zp},format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{dur:.3f}",
        out_mp4
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

async def main():
    audio_file, orig_duration = download_audio_and_get_duration(VIDEO_URL)
    transcript = transcribe_audio(audio_file)
    story = rewrite_script(transcript, orig_duration)
    character_bible = story.get("character_bible", "")
    scenes = story["scenes"]

    concat_list_path = "temp/concat.txt"
    with open(concat_list_path, "w") as cl:
        for idx, scene in enumerate(scenes):
            v_path = f"temp/v_{idx}.mp3"
            img_path = f"temp/img_{idx}.jpg"
            scene_mp4 = f"temp/scene_{idx}.mp4"

            await generate_voice(scene["voice_hindi"], v_path)
            dur = get_audio_duration(v_path)
            generate_consistent_3d_image(character_bible, scene["visual_prompt"], img_path, idx)
            render_scene_with_ffmpeg(img_path, v_path, scene_mp4, dur, idx)
            cl.write(f"file 'scene_{idx}.mp4'\n")

    out_file = "final_reel.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", out_file],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    final_dur = get_audio_duration(out_file)
    with open(out_file, "rb") as video:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo",
            data={
                "chat_id": CHAT_ID,
                "caption": f"✅ Clean 3D Pixar Story Reel ({round(final_dur)}s — Original {round(orig_duration)}s)"
            },
            files={"video": video}
        )

if __name__ == "__main__":
    asyncio.run(main())
