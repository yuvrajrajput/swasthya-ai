# Swasthya AI

A Streamlit chat app where users describe health symptoms in Hindi or Hinglish and receive friendly Hindi replies from Claude. Supports multi-turn chat, emergency keyword alerts, and optional voice input.

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. ffmpeg is required for voice input. Install it:
   - Windows: `winget install ffmpeg`
   - Mac: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

3. Copy `.env.example` to `.env` and add your [Anthropic API key](https://console.anthropic.com/):

```
ANTHROPIC_API_KEY=sk-ant-...
```

4. Run the app:

```bash
streamlit run app.py
```

## Notes

- Uses model `claude-sonnet-4-20250514`.
- Multi-turn chat with `st.session_state` history sent to Claude each turn.
- Emergency keywords (e.g. chest pain, seene mein dard, behoshi) trigger an immediate Hindi alert to call **112** before Claude responds.
- Voice: `st.audio_input` (WebM from browser) → pydub/FFmpeg → WAV → Google speech recognition (Hindi); falls back to a “coming soon” note on older Streamlit.
- This app provides general information only, not medical diagnosis.
