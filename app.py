import io
import os

import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-20250514"
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


def get_client() -> Anthropic | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict[str, str | bool]] = []
    if "last_audio_id" not in st.session_state:
        st.session_state.last_audio_id = None


def detect_emergency(text: str) -> bool:
    normalized = text.lower()
    return any(keyword in normalized for keyword in EMERGENCY_KEYWORDS)


def show_emergency_warning() -> None:
    st.markdown(
        f'<div style="background-color:#b91c1c;color:#fff;padding:1rem 1.25rem;'
        f'border-radius:0.5rem;margin:0.5rem 0;line-height:1.6;">'
        f"{EMERGENCY_WARNING_HTML}</div>",
        unsafe_allow_html=True,
    )


def ask_claude(history: list[dict[str, str | bool]]) -> str:
    client = get_client()
    if client is None:
        raise ValueError("API key not configured")

    api_messages = [
        {"role": m["role"], "content": str(m["content"])}
        for m in history
        if m["role"] in ("user", "assistant")
    ]

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=api_messages,
    )
    return message.content[0].text


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


def process_user_message(user_text: str, from_voice: bool = False) -> None:
    user_text = user_text.strip()
    if not user_text:
        return

    display_text = f"*(आवाज़ से)* {user_text}" if from_voice else user_text
    is_emergency = detect_emergency(user_text)

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

    with st.spinner("विचार किया जा रहा है..."):
        try:
            response = ask_claude(st.session_state.messages)
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "emergency": False}
            )
        except Exception as e:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"कुछ गलत हो गया। बाद में पुनः प्रयास करें। ({e})",
                    "emergency": False,
                }
            )

    st.rerun()


def handle_voice_input() -> None:
    st.markdown("**Ya bolkar likhaiye**")
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
    st.set_page_config(
        page_title="Swasthya AI",
        page_icon="🩺",
        layout="centered",
    )

    init_session_state()

    st.title("Swasthya AI")
    st.caption(
        "हिंदी या Hinglish में बात करें — सरल सुझाव पाएं "
        "(जैसे: mujhe bukhaar hai aur headache ho raha hai)"
    )

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error(
            "Claude API key सेट नहीं है। `.env` में `ANTHROPIC_API_KEY` जोड़ें "
            "(`.env.example` देखें)।"
        )
        st.stop()

    _, col2 = st.columns([4, 1])
    with col2:
        if st.button("नई बातचीत", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_audio_id = None
            st.rerun()

    render_chat_history()

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
