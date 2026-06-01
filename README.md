# Swasthya AI

A Streamlit health assistant for India. Users describe symptoms in **Hindi or Hinglish** and get friendly **Hindi** replies from Claude. Built for phones and low-bandwidth users (village to city).

**Live app:** [swasthya-ai.streamlit.app](https://swasthya-ai.streamlit.app)  
**Landing page:** `index.html` (GitHub Pages or static host)

## Features (Milestone 2)

| Feature | Details |
|--------|---------|
| **Claude** | `claude-sonnet-4-6` via LangChain `ChatAnthropic` |
| **Streaming** | Responses stream token-by-token (`st.write_stream`) |
| **Memory** | Last **3** user/assistant turns sent per request (token savings) |
| **Semantic cache** | `sentence-transformers` cosine similarity (≥ 0.88); first turn only; never for emergencies |
| **Logging** | Supabase `query_logs` — latency, tokens, cost, `cached` flag (optional) |
| **Cache storage** | Supabase `response_cache` with RLS policies (`supabase_setup.sql`) |
| **Emergency routing** | Hindi/English keywords → **112** alert before Claude |
| **Voice** | `streamlit-mic-recorder` → Google Speech (`hi-IN`); mic above chat input; best on **Chrome Android** |
| **i18n** | Hindi system prompt; accepts Hinglish input |

## Project files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit chat app |
| `index.html` | Marketing / waitlist landing |
| `requirements.txt` | Python dependencies |
| `supabase_setup.sql` | RLS policies (run after creating tables) |
| `.env.example` | Local env template |
| `.streamlit/config.toml` | Saffron/cream theme |

## Setup (local)

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in:

```
ANTHROPIC_API_KEY=sk-ant-...
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=sb_publishable_...
```

`SUPABASE_*` is optional — without it, logging and cache are skipped (app still works).

3. Run the app:

```bash
python -m streamlit run app.py
```

First run downloads the embedding model (~100MB) for semantic cache — one-time.

## Supabase setup

1. Create a project at [supabase.com](https://supabase.com).
2. In **SQL Editor**, create tables (`query_logs`, `response_cache`) — see project history or your saved SQL.
3. Run all SQL from **`supabase_setup.sql`** (RLS + insert/select policies).
4. In **Settings → API**, copy **Project URL** and **Publishable** key (not the secret key).
5. Add to `.env` and Streamlit Cloud Secrets (see below).

## Deploy on Streamlit Cloud

1. Push latest code to **`main`**: [github.com/yuvrajrajput/swasthya-ai](https://github.com/yuvrajrajput/swasthya-ai)
2. Open [share.streamlit.io](https://share.streamlit.io) → **Create app** (or open existing)
3. **Branch:** `main` · **Main file:** `app.py`
4. **Secrets** (Settings → Secrets):

```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "sb_publishable_your-key-here"
```

5. **Deploy** or **Reboot** after each `main` update.

**Cloud notes:**
- Do not commit `.env` — use Secrets only.
- First deploy / cold start can take several minutes (embedding model + dependencies).
- Voice needs **HTTPS**, **mic permission**, and **internet** (Google STT). Recommend **Chrome** on Android.
- **ffmpeg is not required** for the current voice stack.

## Voice (how it works)

1. Tap **🎤 बोलें** → speak in Hindi/Hinglish → tap **⏹ रोकें**.
2. Audio is transcribed server-side (`SpeechRecognition` + Google, language `hi-IN`).
3. Text is sent to chat as `*(आवाज़ से)* …` and Claude replies (streamed).

If voice fails, type symptoms in the box at the bottom.

## Architecture notes

- **No medical diagnosis** — general information only; disclaimer shown in-app.
- **Cache:** Similar first messages reuse a stored reply (cosine on multilingual embeddings); follow-up messages in the same chat are not cached.
- **Cost:** ~$0.01 per Claude call when cache misses; cached hits cost $0.
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (loaded once, `@st.cache_resource`).

## License

See `LICENSE`.
