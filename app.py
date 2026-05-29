import io
import json
import os
import re
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

MODEL = "claude-sonnet-4-6"
LOGO_PATH = "swasthya-ai-icon.svg"
QUERY_LOGS_PATH = "query_logs.json"
RESPONSE_CACHE_PATH = "response_cache.json"
MAX_HISTORY_TURNS = 3
CACHE_SIMILARITY_THRESHOLD = 0.88
INPUT_COST_PER_MTOK = 3.0
OUTPUT_COST_PER_MTOK = 15.0
HAS_AUDIO_INPUT = hasattr(st, "audio_input")

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


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict[str, str | bool]] = []
    if "last_audio_id" not in st.session_state:
        st.session_state.last_audio_id = None


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


def _load_json_list(path: str) -> list:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_json_list(path: str, items: list) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except OSError:
        pass


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
    entry = {
        "query": normalized,
        "length": query_token_count(normalized),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "was_emergency": was_emergency,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        "cached": cached,
    }
    logs = _load_json_list(QUERY_LOGS_PATH)
    logs.append(entry)
    _save_json_list(QUERY_LOGS_PATH, logs)


def load_response_cache() -> list[dict]:
    return _load_json_list(RESPONSE_CACHE_PATH)


def save_to_response_cache(normalized_query: str, response: str) -> None:
    cache = load_response_cache()
    cache = [e for e in cache if e.get("query") != normalized_query]
    cache.append(
        {
            "query": normalized_query,
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    if len(cache) > 500:
        cache = cache[-500:]
    _save_json_list(RESPONSE_CACHE_PATH, cache)


def find_cached_response(normalized_query: str) -> str | None:
    """Semantic-lite cache: exact match, then high similarity on normalized text."""
    for entry in load_response_cache():
        if entry.get("query") == normalized_query:
            return str(entry.get("response", ""))
    best_ratio = 0.0
    best_response: str | None = None
    for entry in load_response_cache():
        cached_q = str(entry.get("query", ""))
        ratio = SequenceMatcher(None, normalized_query, cached_q).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_response = str(entry.get("response", ""))
    if best_ratio >= CACHE_SIMILARITY_THRESHOLD and best_response:
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


def ask_claude(history: list[dict[str, str | bool]]) -> tuple[str, dict]:
    """LangChain ChatAnthropic invoke; returns (text, metrics)."""
    llm = get_llm()
    lc_messages = build_langchain_messages(history)

    start = time.perf_counter()
    ai_message = llm.invoke(lc_messages)
    latency_ms = int((time.perf_counter() - start) * 1000)

    raw_content = ai_message.content
    if isinstance(raw_content, str):
        text = raw_content
    elif isinstance(raw_content, list):
        parts = []
        for block in raw_content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        text = "".join(parts) or str(raw_content)
    else:
        text = str(raw_content)

    usage = getattr(ai_message, "usage_metadata", None) or {}
    if not usage.get("input_tokens"):
        meta = getattr(ai_message, "response_metadata", None) or {}
        usage = meta.get("usage", usage) or usage
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    cost_usd = estimate_cost_usd(input_tokens, output_tokens)

    metrics = {
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "cached": False,
    }
    return text, metrics


def show_emergency_warning() -> None:
    st.markdown(
        f'<div style="background-color:#b91c1c;color:#fff;padding:1rem 1.25rem;'
        f'border-radius:0.5rem;margin:0.5rem 0;line-height:1.6;">'
        f"{EMERGENCY_WARNING_HTML}</div>",
        unsafe_allow_html=True,
    )


def transcribe_audio(audio_bytes: bytes) -> str | None:
    try:
        import speech_recognition as sr
    except ImportError:
        return None

    recognizer = sr.Recognizer()
    try:
        from pydub import AudioSegment

        audio_segment = AudioSegment.from_file(
            io.BytesIO(audio_bytes), format="webm"
        )
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        with sr.AudioFile(wav_buffer) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio, language="hi-IN")
    except Exception:
        return None


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

    # Cache only for non-emergency, first symptom-style turn (not multi-turn follow-ups)
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

    with st.spinner("विचार किया जा रहा है..."):
        try:
            response, metrics = ask_claude(st.session_state.messages)
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "emergency": False}
            )
            if use_cache:
                save_to_response_cache(normalized, response)
            log_query_metrics(
                user_text,
                is_emergency,
                latency_ms=metrics["latency_ms"],
                input_tokens=metrics["input_tokens"],
                output_tokens=metrics["output_tokens"],
                cost_usd=metrics["cost_usd"],
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
          [data-testid="stChatMessage"] { line-height: 1.65; }
          [data-testid="stChatMessage"] h1,
          [data-testid="stChatMessage"] h2,
          [data-testid="stChatMessage"] h3 {
            font-size: 1.05rem !important;
            margin-top: 0.4rem;
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
            st.session_state.last_audio_id = None
            st.rerun()


def handle_voice_input() -> None:
    st.markdown("**🎤 बोलकर बताइए**")
    if not HAS_AUDIO_INPUT:
        st.caption("🎤 आवाज़ से लिखने की सुविधा जल्द आएगी। अभी नीचे टाइप करें।")
        return

    audio = st.audio_input(
        "माइक्रोफ़ोन दबाएं और बोलें",
        key="voice_input",
    )
    if audio is None:
        return

    audio_bytes = audio.getvalue()
    audio_id = hash(audio_bytes)
    if st.session_state.last_audio_id == audio_id:
        return

    st.session_state.last_audio_id = audio_id
    with st.spinner("आवाज़ समझी जा रही है..."):
        transcript = transcribe_audio(audio_bytes)

    if transcript:
        process_user_message(transcript, from_voice=True)
    else:
        st.warning(
            "आवाज़ समझ नहीं आई। कृपया फिर बोलें या नीचे टाइप करें। "
            "(स्पष्ट हिंदी/हिंग्लिश में बोलें)"
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

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error(
            "Claude API key सेट नहीं है। "
            "**Streamlit Cloud:** Settings → Secrets → `ANTHROPIC_API_KEY = \"sk-ant-...\"` → Reboot. "
            "**Local:** `.env` में `ANTHROPIC_API_KEY` जोड़ें (`.env.example` देखें)।"
        )
        st.stop()

    render_header()
    render_chat_history()

    st.divider()
    handle_voice_input()

    prompt = st.chat_input("अपने लक्षण लिखें...")
    if prompt:
        process_user_message(prompt)

    st.divider()
    st.caption(
        "⚠️ यह ऐप चिकित्सा निदान नहीं देता। गंभीर या लंबे समय से चल रहे "
        "लक्षणों के लिए हमेशा योग्य डॉक्टर से परामर्श लें।"
    )


if __name__ == "__main__":
    main()
