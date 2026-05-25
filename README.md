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

## Deploy on Streamlit Cloud

1. Merge latest code to `main` on GitHub: [yuvrajrajput/swasthya-ai](https://github.com/yuvrajrajput/swasthya-ai)
2. Open **[share.streamlit.io](https://share.streamlit.io)** → sign in with **GitHub**
3. Click **Create app** → pick repository **`swasthya-ai`**
4. Set:
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Advanced settings** → **Secrets** and paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

(Get your key from [console.anthropic.com](https://console.anthropic.com/).)

6. Click **Deploy**

Your app URL will look like: `https://swasthya-ai-xxxxx.streamlit.app`

**Notes for Cloud:**
- `packages.txt` installs **ffmpeg** for voice input on the server
- Do not commit `.env` — use Streamlit **Secrets** only
- First deploy may take 2–5 minutes
- Link this URL from your landing page **Try Now** button when ready

## Notes

- Uses model `claude-sonnet-4-6` (Claude Sonnet 4.6).
- Multi-turn chat with `st.session_state` history sent to Claude each turn.
- Emergency keywords (e.g. chest pain, seene mein dard, behoshi) trigger an immediate Hindi alert to call **112** before Claude responds.
- Voice: `st.audio_input` (WebM from browser) → pydub/FFmpeg → WAV → Google speech recognition (Hindi); falls back to a “coming soon” note on older Streamlit.
- This app provides general information only, not medical diagnosis.
