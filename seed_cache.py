import os
import re
import time
from datetime import datetime, timezone

from anthropic import Anthropic
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MODEL = "claude-sonnet-4-6"

SEED_QUERIES = [
    "mujhe bukhaar hai",
    "sar dard ho raha hai",
    "pet mein dard hai",
    "khansi aa rahi hai",
    "sardi jukam hai",
    "ulti ho rahi hai",
    "dast lag rahe hain",
    "body mein dard hai",
    "bahut kamzori lag rahi hai",
    "neend nahi aa rahi",
]

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


def normalize_query(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def generate_response(client: Anthropic, query: str) -> str:
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"मेरे लक्षण हैं:\n\n{query}",
            }
        ],
    )
    return message.content[0].text


def store_in_supabase(sb, normalized_query: str, response: str) -> bool:
    try:
        sb.table("response_cache").upsert(
            {
                "query": normalized_query,
                "response": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="query",
        ).execute()
        return True
    except Exception as e:
        print(f"  [X] Supabase error: {e}")
        return False


def main() -> None:
    print("Swasthya AI - Cache Seeder")
    print("=" * 40)

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY missing")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL or SUPABASE_KEY missing")

    anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    success = 0
    failed = 0

    for i, query in enumerate(SEED_QUERIES, 1):
        normalized = normalize_query(query)
        print(f"\n[{i}/{len(SEED_QUERIES)}] Query: {query}")

        try:
            print("  -> Generating response...")
            response = generate_response(anthropic, query)
            print(f"  -> Response: {len(response)} chars")

            stored = store_in_supabase(sb, normalized, response)
            if stored:
                print("  [OK] Stored in Supabase cache")
                success += 1
            else:
                failed += 1

        except Exception as e:
            print(f"  [X] Error: {e}")
            failed += 1

        if i < len(SEED_QUERIES):
            time.sleep(1)

    print("\n" + "=" * 40)
    print(f"[OK] Success: {success}/{len(SEED_QUERIES)}")
    print(f"[X] Failed:  {failed}/{len(SEED_QUERIES)}")
    print("\nCache seeded. First users get faster responses.")


if __name__ == "__main__":
    main()
