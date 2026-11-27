import os
import json
import asyncio
import requests
import yt_dlp
import edge_tts
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

GROQ_API_KEY = os.environ["GROQ_API_KEY"].strip()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"].strip()
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"].strip()
VIDEO_URL = os.environ["VIDEO_URL"].strip()

os.makedirs("temp", exist_ok=True)
CHARACTER_STYLE = "3D Pixar animation style, friendly Indian young man wearing a navy blue hoodie, expressive face, vibrant studio lighting, 9:16 vertical aspect ratio, high detail"

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
    Rewrite this transcript into a 35-second viral Hinglish faceless video script with 4 scenes.
    Return ONLY valid JSON: {{"scenes": [{{"narration": "...", "visual_action": "..."}}]}}
    Transcript: {transcript}
    """
    for model in ["gemini-3.8-flash"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}).json()
        if "candidates" in res:
            raw = res["candidates"][0]["content"]["parts"][0]["text"].replace("```json", "").replace("```", "").strip()
            return json.loads(raw)["scenes"]
    raise Exception(f"Gemini API Error: {res}")

async def generate_voice(text, path):
    await edge_tts.Communicate(text, "hi-IN-MadhurNeural", rate="+10%").save(path)

def generate_image(action, path):
    prompt = requests.utils.quote(f"{CHARACTER_STYLE}, {action}")
    img = requests.get(f"https://image.pollinations.ai/prompt/{prompt}?width=720&height=1280&nologo=true&seed=42").content
    with open(path, "wb") as f:
        f.write(img)

async def main():
    audio_file = download_audio(VIDEO_URL)
    transcript = transcribe_audio(audio_file)
    scenes = rewrite_script(transcript)

    clips = []
    for idx, scene in enumerate(scenes):
        v_path, i_path = f"temp/v_{idx}.mp3", f"temp/i_{idx}.jpg"
        await generate_voice(scene["narration"], v_path)
        generate_image(scene["visual_action"], i_path)

        a_clip = AudioFileClip(v_path)
        i_clip = ImageClip(i_path).with_duration(a_clip.duration).resized(lambda t: 1 + 0.04 * t).with_position(('center', 'center'))
        clips.append(i_clip.with_audio(a_clip))

    final = concatenate_videoclips(clips, method="compose")
    out_file = "final_reel.mp4"
    final.write_videofile(out_file, fps=24, codec="libx264", audio_codec="aac", logger=None)

    with open(out_file, "rb") as video:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo",
            data={"chat_id": CHAT_ID, "caption": "✅ 100% Cloud-Rendered Faceless Video Ready!"},
            files={"video": video}
        )

if __name__ == "__main__":
    asyncio.run(main())
