import os
import io
import json
import time
import textwrap
import asyncio
import requests
import yt_dlp
import edge_tts
from PIL import Image, ImageDraw, ImageFont, ImageStat
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

GROQ_API_KEY = os.environ["GROQ_API_KEY"].strip()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"].strip()
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"].strip()
VIDEO_URL = os.environ["VIDEO_URL"].strip()

os.makedirs("temp", exist_ok=True)

ANIMATION_STYLE = (
    "3D Pixar and Disney animation movie still, vibrant cinematic lighting, "
    "rich detailed full-bleed background environment, expressive 3D characters, "
    "vertical 9:16 composition, Unreal Engine 5 render, 8k, no text, no watermark"
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
    Rewrite this story/transcript into a gripping 35-second viral Hindi/Hinglish animated story reel with 7 short scenes (each scene 1 short sentence).
    
    CRITICAL RULES:
    1. 'voice_hindi': Write in natural Devanagari Hindi script with Hindi words for ALL numbers (e.g., write "बीस साल" instead of "20 साल", "सात हज़ार" instead of "7000+", "दो रेत के बोरे" instead of "2 sandy bags") so text-to-speech pronounces every word naturally like a human storyteller.
    2. 'caption_hinglish': Short punchy Hinglish subtitle (max 8-10 words) to display on screen.
    3. 'visual_prompt': Detailed English scene description for a 3D Pixar animated movie frame. Keep character appearances consistent across all scenes (e.g., "a middle-aged Indian customs officer in khaki uniform with a mustache, and a clever young smuggler wearing a brown jacket pushing a bicycle with sandbags at a border gate"). Describe exact action, camera angle, and full background.

    Return ONLY valid JSON:
    {{"scenes": [{{"voice_hindi": "...", "caption_hinglish": "...", "visual_prompt": "..."}}]}}
    
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
    # Natural storytelling voice in Hindi
    await edge_tts.Communicate(text, "hi-IN-MadhurNeural", rate="+8%", pitch="+0Hz").save(path)

def add_bold_subtitle(img, text):
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
    except Exception:
        font = ImageFont.load_default()

    lines = textwrap.wrap(text.upper(), width=24)
    line_height = 54
    total_text_h = len(lines) * line_height
    start_y = h - total_text_h - 160

    # Semi-transparent dark pill behind text for high readability
    draw.rounded_rectangle(
        [40, start_y - 20, w - 40, start_y + total_text_h + 20],
        radius=24,
        fill=(0, 0, 0, 175)
    )

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (w - text_w) // 2
        y = start_y + i * line_height
        # Black outline + bright yellow/white text
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
        draw.text((x, y), line, font=font, fill=(255, 230, 50, 255))

    return img

def generate_animated_frame(visual_prompt, caption, out_path, idx):
    full_prompt = f"{visual_prompt}, {ANIMATION_STYLE}"
    encoded = requests.utils.quote(full_prompt)
    urls = [
        f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1360&model=flux&nologo=true&enhance=true&seed={777 + idx * 13}",
        f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1360&nologo=true&seed={888 + idx * 13}"
    ]

    img = None
    for url in urls:
        for _ in range(3):
            try:
                res = requests.get(url, timeout=60)
                if res.status_code == 200 and len(res.content) > 25000:
                    candidate = Image.open(io.BytesIO(res.content)).convert("RGB")
                    # Verify image is not blank white or pure black
                    mean_brightness = sum(ImageStat.Stat(candidate).mean) / 3
                    if 15 < mean_brightness < 245:
                        w, h = candidate.size
                        img = candidate.crop((0, 0, w, int(h * 0.93))).resize((720, 1280), Image.Resampling.LANCZOS)
                        time.sleep(4)
                        break
            except Exception:
                pass
            time.sleep(5)
        if img is not None:
            break

    if img is None:
        img = Image.new("RGB", (720, 1280), color=(24, 32, 56))

    img = add_bold_subtitle(img, caption)
    img.save(out_path, "JPEG", quality=95)

async def main():
    audio_file = download_audio(VIDEO_URL)
    transcript = transcribe_audio(audio_file)
    scenes = rewrite_script(transcript)

    clips = []
    for idx, scene in enumerate(scenes):
        v_path = f"temp/v_{idx}.mp3"
        i_path = f"temp/i_{idx}.jpg"
        await generate_voice(scene["voice_hindi"], v_path)
        generate_animated_frame(scene["visual_prompt"], scene["caption_hinglish"], i_path, idx)

        a_clip = AudioFileClip(v_path)
        dur = max(a_clip.duration, 1.5)

        # Dynamic cinematic motion (alternating zoom-in & zoom-out so every scene feels alive)
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
            data={"chat_id": CHAT_ID, "caption": "✅ 3D Animated Story Reel + Bold Captions Ready!"},
            files={"video": video}
        )

if __name__ == "__main__":
    asyncio.run(main())
