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
from PIL import Image, ImageDraw, ImageFont, ImageStat

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
    return audio_path, max(15.0, min(60.0, duration))

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
    num_scenes = max(5, min(11, round(target_duration / 4.2)))
    words_per_scene = max(7, round((target_duration * 2.3) / num_scenes))

    prompt = f"""
    You are an elite viral Hindi Instagram Reels creator. Rewrite this transcript into a gripping {round(target_duration)}-second Hindi storytelling reel with {num_scenes} fast-paced scenes.

    STRICT RULES:
    1. 'hook_title_hinglish': A 4-to-6 word punchy hook title in ROMAN ENGLISH/HINGLISH ONLY (e.g., "RAVAN KI LANKA KA SACH" or "YEH RAHASYA AAPKO HILA DEGA") to flash for the first 3 seconds. NO Devanagari here.
    2. 'character_and_world': Define ONE consistent 3D Pixar-style visual world & main character in English (e.g., "3D Pixar animated Indian sage and golden Vedic palace of Lanka, dramatic warm sunset lighting").
    3. 'scenes': Array of {num_scenes} scenes. Each scene must have:
       - 'voice_hindi': Natural, emotional spoken Hindi in Devanagari. Write EVERY number in spoken Hindi words (e.g., "पाँच हज़ार", "बीस साल"). Keep ~{words_per_scene} words.
       - 'emotion': Either "hook" (energetic), "narrator" (deep storytelling), or "twist" (dramatic/intense).
       - 'visual_prompt': Single cohesive scene action in English (MAX 18 words) matching 'character_and_world', specifying camera shot (wide shot / close-up / aerial view).

    Return ONLY valid JSON:
    {{"hook_title_hinglish": "...", "character_and_world": "...", "scenes": [{{"voice_hindi": "...", "emotion": "...", "visual_prompt": "..."}}]}}

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

async def generate_voice(text, emotion, path):
    clean_text = text.strip() or "कहानी आगे बढ़ती है।"
    # Vary voice tone & pacing based on scene emotion for human-like engagement
    if emotion == "hook":
        voice_order = [("hi-IN-MadhurNeural", "+12%", "+2Hz"), ("hi-IN-SwaraNeural", "+10%", "+2Hz")]
    elif emotion == "twist":
        voice_order = [("hi-IN-SwaraNeural", "+4%", "-1Hz"), ("hi-IN-MadhurNeural", "+2%", "-3Hz")]
    else:
        voice_order = [("hi-IN-MadhurNeural", "+7%", "+0Hz"), ("hi-IN-SwaraNeural", "+6%", "+0Hz")]

    for voice, rate, pitch in voice_order:
        for _ in range(3):
            try:
                await edge_tts.Communicate(clean_text, voice, rate=rate, pitch=pitch).save(path)
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

def burn_opening_hook_badge(img, hook_text):
    # Clean, modern Roman-script hook banner on Scene 1 only (zero box artifacts)
    clean_hook = "".join(c for c in hook_text.upper() if c.isalnum() or c in " ?!'-").strip()
    if not clean_hook:
        return img
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), clean_hook, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 28, 16
    box_w = min(w - 60, tw + pad_x * 2)
    x0 = (w - box_w) // 2
    y0 = 130

    draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + th + pad_y * 2], radius=18, fill=(15, 15, 20, 210), outline=(255, 210, 60, 255), width=3)
    tx = (w - tw) // 2
    ty = y0 + pad_y
    draw.text((tx, ty), clean_hook, font=font, fill=(255, 230, 80, 255))
    return img

def generate_consistent_3d_image(world_style, visual_prompt, out_jpg, idx, hook_title=""):
    full_prompt = (
        f"3D Pixar Disney animated movie frame, single full-bleed scene, {visual_prompt}, "
        f"{world_style}, rich volumetric golden lighting, 9:16 vertical, "
        f"no split screen, no collage, no borders, no text, no watermark"
    )

    img = None
    if HF_TOKEN:
        hf_models = ["black-forest-labs/FLUX.1-schnell", "stabilityai/stable-diffusion-xl-base-1.0"]
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
                        img = cand.resize((720, 1280), Image.Resampling.LANCZOS)
                        print(f"Scene {idx+1}: HuggingFace {hf_m} rendered!")
                        break
            except Exception:
                pass

    if img is None:
        encoded = requests.utils.quote(full_prompt[:220])
        for attempt in range(6):
            model_name = "flux" if attempt % 2 == 0 else "turbo"
            url = f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1360&model={model_name}&nologo=true&seed={random.randint(1000, 99999)}"
            try:
                res = requests.get(url, timeout=60)
                if res.status_code == 200 and len(res.content) > 30000:
                    cand = Image.open(io.BytesIO(res.content)).convert("RGB")
                    if is_clean_image(cand):
                        w, h = cand.size
                        img = cand.crop((0, 0, w, int(h * 0.93))).resize((720, 1280), Image.Resampling.LANCZOS)
                        print(f"Scene {idx+1}: Pollinations rendered!")
                        time.sleep(15)
                        break
            except Exception:
                pass
            time.sleep(15)

    if img is None:
        img = Image.new("RGB", (720, 1280), color=(25, 35, 65))
        draw = ImageDraw.Draw(img)
        for y in range(1280):
            draw.line([(0, y), (720, y)], fill=(20 + int(35 * y / 1280), 30 + int(45 * y / 1280), 65 + int(75 * y / 1280)))

    if idx == 0 and hook_title:
        img = burn_opening_hook_badge(img, hook_title)

    img.save(out_jpg, "JPEG", quality=95)

def render_scene_with_ffmpeg(img_path, audio_path, out_mp4, dur, idx):
    frames = int(dur * 30)
    mode = idx % 3
    if mode == 0:
        zp = f"zoompan=z='min(zoom+0.0018,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=720x1280:fps=30"
    elif mode == 1:
        zp = f"zoompan=z='if(lte(zoom,1.0),1.15,max(1.001,zoom-0.0018))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=720x1280:fps=30"
    else:
        zp = f"zoompan=z='1.12':x='(iw-iw/zoom)*(on/{max(frames,1)})':y='ih/2-(ih/zoom/2)':d={frames}:s=720x1280:fps=30"

    # Subtle fade-in/out on video edges for polished transitions
    fade_out_start = max(0.1, dur - 0.25)
    vf = f"scale=1440:2560,{zp},fade=t=in:st=0:d=0.2,fade=t=out:st={fade_out_start:.2f}:d=0.2,format=yuv420p"

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", img_path,
        "-i", audio_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{dur:.3f}",
        out_mp4
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def add_cinematic_bgm(raw_video, final_video, total_dur):
    bgm_wav = "temp/bgm.wav"
    bgm_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"aevalsrc=0.35*sin(2*PI*110*t)+0.25*sin(2*PI*165*t)+0.20*sin(2*PI*220*t)+0.15*sin(2*PI*55*t):s=44100:d={total_dur:.2f}",
        "-af", "lowpass=f=420,aecho=0.8:0.88:60:0.4,volume=0.14",
        bgm_wav
    ]
    subprocess.run(bgm_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    mix_cmd = [
        "ffmpeg", "-y",
        "-i", raw_video,
        "-i", bgm_wav,
        "-filter_complex", "[0:a]volume=1.15[v];[1:a]volume=1.0[b];[v][b]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        final_video
    ]
    subprocess.run(mix_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

async def main():
    audio_file, orig_duration = download_audio_and_get_duration(VIDEO_URL)
    transcript = transcribe_audio(audio_file)
    story = rewrite_script(transcript, orig_duration)
    world_style = story.get("character_and_world", "3D Pixar animated Indian epic cinematic world")
    hook_title = story.get("hook_title_hinglish", "")
    scenes = story["scenes"]

    concat_list_path = "temp/concat.txt"
    with open(concat_list_path, "w") as cl:
        for idx, scene in enumerate(scenes):
            v_path = f"temp/v_{idx}.mp3"
            img_path = f"temp/img_{idx}.jpg"
            scene_mp4 = f"temp/scene_{idx}.mp4"

            await generate_voice(scene["voice_hindi"], scene.get("emotion", "narrator"), v_path)
            dur = get_audio_duration(v_path)
            generate_consistent_3d_image(world_style, scene["visual_prompt"], img_path, idx, hook_title if idx == 0 else "")
            render_scene_with_ffmpeg(img_path, v_path, scene_mp4, dur, idx)
            cl.write(f"file 'scene_{idx}.mp4'\n")

    raw_out = "temp/raw_final.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", raw_out],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    out_file = "final_reel.mp4"
    final_dur = get_audio_duration(raw_out)
    add_cinematic_bgm(raw_out, out_file, final_dur)

    with open(out_file, "rb") as video:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo",
            data={
                "chat_id": CHAT_ID,
                "caption": f"✅ Studio-Polished 3D Animated Reel ({round(final_dur)}s — Original {round(orig_duration)}s)"
            },
            files={"video": video}
        )

if __name__ == "__main__":
    asyncio.run(main())