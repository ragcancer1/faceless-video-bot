name: Render Faceless Video
on:
  workflow_dispatch:
    inputs:
      video_url:
        description: 'Social Media Video URL'
        required: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          sudo apt-get update && sudo apt-get install -y ffmpeg
          pip install yt-dlp edge-tts moviepy requests pillow
      - name: Generate Video
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          PIXABAY_API_KEY: ${{ secrets.PIXABAY_API_KEY }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          VIDEO_URL: ${{ github.event.inputs.video_url }}
        run: python make_video.py
