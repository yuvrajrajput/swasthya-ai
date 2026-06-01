import os
import re
import time
from collections.abc import Iterator
from datetime import datetime, timezone
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from streamlit_mic_recorder import speech_to_text
from supabase import create_client

load_dotenv()

MODEL = "claude-sonnet-4-6"
LOGO_PATH = "swasthya-ai-icon.svg"
MAX_HISTORY_TURNS = 3
CACHE_SIMILARITY_THRESHOLD = 0.88
INPUT_COST_PER_MTOK = 3.0
OUTPUT_COST_PER_MTOK = 15.0

EMERGENCY_KEYWORDS = [
    "chest pain",
    "heart attack",
    "seene mein dard",
    "seene me dard",
    "seenay mein dard",
    "seene me",
    "sans lene mein takleef",
    "saans lene mein takleef",
    "sans lene",
    "saans lene",
    "dam ghut",
    "behoshi",
    "behosh",
    "tez bukhar",
    "teez bukhar",
    "khoon",
    "ulti",
    "stroke",
    "सीने में दर्द",
    "सीने में",
    "सांस लेने",
    "बेहोशी",
    "बेहोश",
    "तेज बुखार",
    "खून",
    "उल्टी",
    "स्ट्रोक",
    "दिल का दौरा",
]

EMERGENCY_WARNING_HTML = (
    "<strong>🚨 संभावित आपातकाल — तुरंत कार्रवाई करें</strong><br><br>"
    "आपके लक्षण गंभीर हो सकते हैं। <strong>अभी 112 पर कॉल करें</strong> या "
    "नज़दीकी <strong>आपातकालीन कक्ष (ER)</strong> में जाएं।<br>"
    "इंतज़ार न करें — यह ऐप आपातकालीन इलाज की जगह नहीं ले सकता।"
)

SYSTEM_PROMPT = """आप एक सहायक स्वास्थ्य जानकारी सहायक हैं। उपयोगकर्ता अपने लक्षण बताता है।

भाषा:
- उपयोगकर्ता हिंदी, अंग्रेज़ी, या Hinglish (मिश्रित) में लिख सकता है — जैसे "mujhe bukhaar hai aur headache ho raha hai"
- आपको हमेशा सरल, दोस्ताना **हिंदी** में जवाब देना है (इनपुट की भाषा से स्वतंत्र)
- अंग्रेज़ी शब्दों को समझें पर जवाब में सरल हिंदी प्रयोग करें

आपका काम:
1. सरल हिंदी में जवाब दें (बहुत तकनीकी शब्द न करें)
2. संभावित कारणों का संक्षिप्त विवरण दें — निदान न करें, केवल सामान्य जानकारी दें
3. स्पष्ट बताएं कि क्या डॉक्टर से मिलना चाहिए (हाँ / जल्दी / आपातकाल) और क्यों
4. घर पर सुरक्षित सामान्य सुझाव दें, अगर उपयुक्त हों
5. जवाब 150–250 शब्दों में रखें (बातचीत जारी हो तो संक्षिप्त भी चलेगा)

महत्वपूर्ण:
- यह चिकित्सा निदान नहीं है; हमेशा अंत में एक पंक्ति में लिखें: "यह केवल सामान्य जानकारी है — डॉक्टर की सलाह को प्राथमिकता दें।"
- गंभीर लक्षणों पर तुरंत डॉक्टर या आपातकालीन सेवा जाने की सलाह दें
- दवाओं की खुराक या नुस्खे न बताएं।"""


@st.cache_resource
def get_llm() -> ChatAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API key not configured")
    return ChatAnthropic(
        model=MODEL,
        max_tokens=1024,
        api_key=api_key,
    )


@st.cache_resource(show_spinner="Model load ho raha hai...")
def get_embedder():
    return SentenceTransformer(
        'paraphrase-multilingual-MiniLM-L12-v2'
    )


def _env_or_secret(name: str) -> str | None:
    val = os.getenv(name)
    if val:
        return val
    try:
        return str(st.secrets[name])
    except Exception:
        return None


@st.cache_resource
def get_supabase():
    url = _env_or_secret("SUPABASE_URL")
    key = _env_or_secret("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict[str, str | bool]] = []
    if "_last_voice_text" not in st.session_state:
        st.session_state._last_voice_text = ""


def detect_emergency(text: str) -> bool:
    normalized = text.lower()
    return any(keyword in normalized for keyword in EMERGENCY_KEYWORDS)


def normalize_query(text: str) -> str:
    """Lowercase, collapse whitespace; strip email/phone-like patterns."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",
        "[email]",
        text,
    )
    text = re.sub(r"\b\d{10}\b", "[phone]", text)
    text = re.sub(r"\b\+?\d[\d\s-]{8,14}\d\b", "[phone]", text)
    return text


def query_token_count(text: str) -> int:
    return len(text.split()) if text else 0


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * INPUT_COST_PER_MTOK + (
        output_tokens / 1_000_000
    ) * OUTPUT_COST_PER_MTOK


def get_embedding(text: str) -> np.ndarray:
    model = get_embedder()
    return model.encode(text, normalize_embeddings=True)


def log_query_metrics(
    user_text: str,
    was_emergency: bool,
    *,
    latency_ms: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    cached: bool = False,
) -> None:
    """Anonymous query log — no PII, no assistant response text."""
    normalized = normalize_query(user_text)
    length = query_token_count(normalized)
    sb = get_supabase()
    if sb is None:
        st.session_state["_last_log_error"] = "Supabase URL/KEY missing in Secrets or .env"
        return
    row = {
        "query": normalized,
        "length": length,
        "was_emergency": was_emergency,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        "cached": cached,
    }
    try:
        # returning="minimal" — no SELECT after INSERT (query_logs has no public SELECT policy)
        sb.table("query_logs").insert(row, returning="minimal").execute()
        st.session_state.pop("_last_log_error", None)
    except Exception as e:
        st.session_state["_last_log_error"] = str(e)
        try:
            sb.table("query_logs").insert(
                {
                    "query": normalized,
                    "length": length,
                    "was_emergency": was_emergency,
                },
                returning="minimal",
            ).execute()
            st.session_state.pop("_last_log_error", None)
        except Exception as e2:
            st.session_state["_last_log_error"] = str(e2)


def load_response_cache() -> list[dict]:
    sb = get_supabase()
    if sb is None:
        return []
    try:
        result = (
            sb.table("response_cache")
            .select("query,response,timestamp")
            .execute()
        )
        return result.data or []
    except Exception:
        return []


def save_to_response_cache(normalized_query: str, response: str) -> None:
    sb = get_supabase()
    if sb is None:
        return
    try:
        sb.table("response_cache").upsert(
            {
                "query": normalized_query,
                "response": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="query",
        ).execute()
    except Exception:
        pass


def find_cached_response(normalized_query: str) -> str | None:
    cache = load_response_cache()
    if not cache:
        return None

    query_embedding = get_embedding(normalized_query)

    best_score = 0.0
    best_response: str | None = None

    for entry in cache:
        cached_q = str(entry.get("query", ""))

        if cached_q == normalized_query:
            return str(entry.get("response", ""))

        cached_embedding = get_embedding(cached_q)
        score = float(np.dot(query_embedding, cached_embedding))

        if score > best_score:
            best_score = score
            best_response = str(entry.get("response", ""))

    if best_score >= CACHE_SIMILARITY_THRESHOLD and best_response:
        return best_response

    return None


def build_langchain_messages(
    history: list[dict[str, str | bool]],
) -> list[SystemMessage | HumanMessage | AIMessage]:
    """Last N user/assistant turns only — reduces tokens (LangChain messages)."""
    messages: list[SystemMessage | HumanMessage | AIMessage] = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]
    chat_msgs = [m for m in history if m["role"] in ("user", "assistant")]
    for msg in chat_msgs[-(MAX_HISTORY_TURNS * 2) :]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=str(msg["content"])))
        else:
            messages.append(AIMessage(content=str(msg["content"])))
    return messages


def _chunk_text(chunk) -> str:
    if not hasattr(chunk, "content"):
        return ""
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content else ""


def _usage_from_chunk(chunk) -> tuple[int, int]:
    usage = getattr(chunk, "usage_metadata", None) or {}
    if not usage.get("input_tokens"):
        meta = getattr(chunk, "response_metadata", None) or {}
        usage = meta.get("usage", usage) or usage
    return (
        int(usage.get("input_tokens", 0) or 0),
        int(usage.get("output_tokens", 0) or 0),
    )


def stream_claude(history: list[dict[str, str | bool]]) -> Iterator[str]:
    llm = get_llm()
    lc_messages = build_langchain_messages(history)
    start = time.perf_counter()

    full_text = ""
    input_tokens = 0
    output_tokens = 0

    for chunk in llm.stream(lc_messages):
        text = _chunk_text(chunk)
        if text:
            full_text += text
            yield text

        inp, out = _usage_from_chunk(chunk)
        if inp:
            input_tokens = inp
        if out:
            output_tokens = out

    latency_ms = int((time.perf_counter() - start) * 1000)
    cost_usd = estimate_cost_usd(input_tokens, output_tokens)

    stream_claude.last_result = {
        "full_text": full_text,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "cached": False,
    }


def show_emergency_warning() -> None:
    st.markdown(
        f'<div style="background-color:#b91c1c;color:#fff;padding:1rem 1.25rem;'
        f'border-radius:0.5rem;margin:0.5rem 0;line-height:1.6;">'
        f"{EMERGENCY_WARNING_HTML}</div>",
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(str(msg["role"])):
            st.markdown(str(msg.get("display", msg["content"])))
            if msg.get("emergency"):
                show_emergency_warning()


def count_user_turns(history: list[dict]) -> int:
    return sum(1 for m in history if m["role"] == "user")


def process_user_message(user_text: str, from_voice: bool = False) -> None:
    user_text = user_text.strip()
    if not user_text:
        return

    display_text = f"*(आवाज़ से)* {user_text}" if from_voice else user_text
    is_emergency = detect_emergency(user_text)
    normalized = normalize_query(user_text)

    if is_emergency:
        show_emergency_warning()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_text,
            "display": display_text,
            "emergency": is_emergency,
        }
    )

    use_cache = not is_emergency and count_user_turns(st.session_state.messages) <= 1
    if use_cache:
        cached_response = find_cached_response(normalized)
        if cached_response:
            st.session_state.messages.append(
                {"role": "assistant", "content": cached_response, "emergency": False}
            )
            log_query_metrics(
                user_text,
                is_emergency,
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                cached=True,
            )
            st.rerun()
            return

    try:
        with st.chat_message("assistant"):
            response = st.write_stream(stream_claude(st.session_state.messages))

        metrics = getattr(stream_claude, "last_result", None) or {}
        if not response:
            response = metrics.get("full_text", "")

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "emergency": False}
        )
        if use_cache:
            save_to_response_cache(normalized, response)
        log_query_metrics(
            user_text,
            is_emergency,
            latency_ms=metrics.get("latency_ms", 0),
            input_tokens=metrics.get("input_tokens", 0),
            output_tokens=metrics.get("output_tokens", 0),
            cost_usd=metrics.get("cost_usd", 0.0),
            cached=False,
        )
    except Exception as e:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"कुछ गलत हो गया। बाद में पुनः प्रयास करें। ({e})",
                "emergency": False,
            }
        )
        log_query_metrics(user_text, is_emergency, cached=False)

    st.rerun()


def inject_brand_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp { background-color: #FFF8F3; }
          .block-container { padding-top: 1.25rem; max-width: 42rem; }
          [data-testid="stChatMessage"] {
            line-height: 1.65;
            background: #fff;
            border: 1px solid #f5ebe3;
            border-radius: 12px;
            padding: 0.35rem 0.5rem;
          }
          [data-testid="stChatMessage"] h1,
          [data-testid="stChatMessage"] h2,
          [data-testid="stChatMessage"] h3 {
            font-size: 1.05rem !important;
            margin-top: 0.4rem;
          }
          .voice-dock-label {
            font-size: 0.8rem;
            color: #6b7280;
            margin: 0 0 0.35rem 0.15rem;
          }
          .voice-dock-box {
            background: #fff;
            border: 1px solid #f0e0d6;
            border-radius: 16px;
            padding: 0.65rem 0.85rem 0.5rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 2px 10px rgba(255,107,43,0.06);
          }
          iframe[title="streamlit_mic_recorder.streamlit_mic_recorder"] {
            min-height: 52px;
          }
          [data-testid="stChatInput"] {
            border-radius: 16px !important;
            border: 1px solid #f0e0d6 !important;
            box-shadow: 0 2px 12px rgba(0,0,0,0.04);
          }
          .stButton > button[kind="secondary"] {
            border-radius: 10px;
            border-color: #f0e0d6;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    col_logo, col_title, col_btn = st.columns([1, 5, 2], vertical_alignment="center")
    with col_logo:
        if os.path.isfile(LOGO_PATH):
            st.image(LOGO_PATH, width=48)
        else:
            st.markdown("### 🩺")
    with col_title:
        st.markdown("### Swasthya AI")
        st.caption(
            "हिंदी या Hinglish में बात करें — सरल सुझाव पाएं "
            "(जैसे: mujhe bukhaar hai aur headache ho raha hai)"
        )
    with col_btn:
        if st.button("नई बातचीत", use_container_width=True):
            st.session_state.messages = []
            st.session_state._last_voice_text = ""
            st.rerun()


def render_voice_input_bar() -> None:
    """Mic above chat input — transcript returns to Python (works on mobile)."""
    st.markdown(
        """
        <div class="voice-dock-box">
          <p class="voice-dock-label">🎤 बोलकर बताएं या नीचे लिखें</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    transcript = speech_to_text(
        language="hi-IN",
        start_prompt="🎤 बोलें",
        stop_prompt="⏹ रोकें",
        just_once=True,
        use_container_width=True,
        key="voice_stt",
    )
    st.caption(
        "टैप → हिंदी में बोलें → ⏹ रोकें। इंटरनेट चाहिए। "
        "अगर आवाज़ न समझे तो नीचे टाइप करें।"
    )

    if not transcript:
        return

    text = str(transcript).strip()
    if not text or text == st.session_state._last_voice_text:
        return

    st.session_state._last_voice_text = text
    process_user_message(text, from_voice=True)


def render_empty_chat_hint() -> None:
    if not st.session_state.messages:
        st.markdown(
            """
            <div style="text-align:center;padding:2rem 1rem;color:#6b7280;">
              <p style="font-size:1.05rem;margin-bottom:0.5rem;">👋 नमस्ते!</p>
              <p style="font-size:0.9rem;line-height:1.5;">
                लक्षण बताएं — हिंदी या Hinglish में।<br>
                <strong>माइक</strong> दबाएं या नीचे <strong>टाइप</strong> करें।
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    page_icon = LOGO_PATH if os.path.isfile(LOGO_PATH) else "🩺"
    st.set_page_config(
        page_title="Swasthya AI",
        page_icon=page_icon,
        layout="centered",
    )

    init_session_state()
    inject_brand_styles()
    render_header()

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error(
            "Claude API key सेट नहीं है। "
            "`.env` में `ANTHROPIC_API_KEY` जोड़ें।"
        )
        st.stop()

    chat_zone = st.container()
    with chat_zone:
        render_empty_chat_hint()
        render_chat_history()

    input_zone = st.container()
    with input_zone:
        render_voice_input_bar()
        prompt = st.chat_input("अपने लक्षण लिखें…")
        if prompt:
            process_user_message(prompt)

    st.caption(
        "⚠️ यह ऐप चिकित्सा निदान नहीं देता। गंभीर लक्षणों के लिए डॉक्टर से मिलें।"
    )
    log_err = st.session_state.get("_last_log_error")
    if log_err:
        st.warning(f"Query log (Supabase): {log_err}")


if __name__ == "__main__":
    main()
