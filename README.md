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
| **Voice** | `streamlit-mic-recorder` → **Google STT** (`hi-IN`, server-side). **Not** browser Web Speech API. Best on **Chrome Android**. |
| **i18n** | Hindi system prompt; accepts Hinglish input |

## Milestone 3 progress

| # | Task | Status |
|---|------|--------|
| 1 | Voice (`streamlit-mic-recorder` + Google STT) | **Done** |
| 2 | Supabase RLS (tables + policies) | Run `supabase_setup.sql` in dashboard → verify below |
| 3 | README accurate | **Done** |
| 4 | Latest code on `main` | **Done** |
| 5 | 50 users + Google Form feedback | **You** (marketing — not code) |
| 6 | Top 10 symptoms from `query_logs` | After traffic — SQL in `supabase_setup.sql` |

## Project files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit chat app |
| `index.html` | Marketing / waitlist landing |
| `requirements.txt` | Python dependencies |
| `supabase_setup.sql` | Tables + RLS policies (run once in SQL Editor) |
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
2. **SQL Editor** → paste and run **`supabase_setup.sql`** (creates tables + enables RLS + policies).
3. **Verify RLS:** Table Editor → `query_logs` / `response_cache` → **RLS enabled** (shield icon).  
   Or run: `select tablename, rowsecurity from pg_tables where tablename in ('query_logs','response_cache');`  
   Both must show `rowsecurity = true`.
4. **Settings → API** → copy **Project URL** + **Publishable** key (not secret key).
5. Add to `.env` and Streamlit Cloud Secrets (see below).

**Security:** With RLS on, the public (anon) key can log queries (insert) and use the cache — but **cannot** read all `query_logs` from the browser. Only you see full logs in the Supabase dashboard.

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

**Stack:** `streamlit-mic-recorder` records audio → Python `SpeechRecognition` → **Google Speech-to-Text** (`hi-IN`).  
This is **not** the browser Web Speech API (that approach was removed — it failed on mobile).

1. Tap **🎤 बोलें** → speak in Hindi/Hinglish → tap **⏹ रोकें**.
2. Wait a few seconds for transcription (needs **internet**).
3. Text appears in chat as `*(आवाज़ से)* …` and Claude replies (streamed).

If voice fails, type symptoms in the box at the bottom. Use **Chrome** on Android for best results.

## Architecture notes

- **No medical diagnosis** — general information only; disclaimer shown in-app.
- **Cache:** Similar first messages reuse a stored reply (cosine on multilingual embeddings); follow-up messages in the same chat are not cached.
- **Cost:** ~$0.01 per Claude call when cache misses; cached hits cost $0.
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (loaded once, `@st.cache_resource`).

## License

See `LICENSE`.
