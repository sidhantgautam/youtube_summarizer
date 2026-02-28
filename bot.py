import os
import re
import ollama
import requests
import string
import numpy as np
import time
import asyncio
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

user_sessions = {}

transcript_cache = {}
MAX_CACHE_SIZE = 10 # prevent memory explosion
MAX_SESSIONS = 50
SESSION_TIMEOUT = 1800  # 30 minutes

STOPWORDS = {
    "the", "is", "are", "was", "were", "what", "when", "where",
    "who", "why", "how", "a", "an", "of", "in", "on", "at",
    "to", "for", "and", "or", "did", "does", "do", "about",
    "this", "that", "it", "he", "she", "they", "we"
}


# ---------------- HELPERS ---------------- #

def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def get_video_title(video_id):
    try:
        url = f"https://www.youtube.com/oembed?url=http://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(url)
        return response.json().get("title", "Unknown Title")
    except Exception:
        return "Unknown Title"

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def clean_text(text):
    """
    Lowercase + remove punctuation.
    """
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def chunk_transcript(text):
    """
    Split transcript into adaptive chunk sizes
    based on total length.
    """

    words = text.split()
    word_count = len(words)

    # ---------- Adaptive Chunk Size ----------
    if word_count < 2000:
        chunk_size = 800
    elif word_count < 6000:
        chunk_size = 1200
    else:
        chunk_size = 1600

    chunks = []

    for i in range(0, word_count, chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    if len(chunks) > 30:
        chunks = chunks[:30]

    return chunks


def is_hindi(text):
    return bool(re.search(r'[\u0900-\u097F]', text))

def detect_transcript_language(transcript_text):
    """
    Ratio-based language detection.
    Returns: 'hi' if significant Hindi content, else 'en'
    """

    if not transcript_text:
        return "en"

    total_chars = len(transcript_text)
    devanagari_chars = len(re.findall(r'[\u0900-\u097F]', transcript_text))

    ratio = devanagari_chars / total_chars

    # If more than 10% Hindi characters → treat as Hindi
    if ratio > 0.10:
        return "hi"

    return "en"

def get_embedding(text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return np.array(response["embedding"])

async def get_embedding_async(text):
    return await asyncio.to_thread(get_embedding, text)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10
    )

def cleanup_sessions():
    current_time = time.time()
    expired = []

    for chat_id, session in user_sessions.items():
        if current_time - session.get("last_active", 0) > SESSION_TIMEOUT:
            expired.append(chat_id)

    for chat_id in expired:
        user_sessions.pop(chat_id, None)

# ---------------- AI FUNCTIONS ---------------- #

def generate_summary(chunks):

    # Step 1 — Summarize each chunk briefly
    chunk_summaries = []

    max_chunks = min(len(chunks), 6)
    for chunk in chunks[:max_chunks]:
        prompt = f"""
Summarize this part of a video transcript in 3 concise sentences.

Transcript:
{chunk}
"""

        response = ollama.chat(
            model='llama3',
            messages=[
                {'role': 'system', 'content': 'Concise summarizer.'},
                {'role': 'user', 'content': prompt}
            ]
        )

        chunk_summaries.append(response['message']['content'])

    combined_summary = "\n".join(chunk_summaries)

    # Step 2 — Generate final structured summary
    final_prompt = f"""
You are a transcript summarizer.

STRICT RULES:
• Plain text only
• Use hyphen (-) for bullet points
• No markdown
• Structured format only

Output EXACTLY:

Video Summary:
2–3 sentence overview.

Key Points:
- Point one
- Point two
- Point three
- Point four
- Point five

Core Takeaway:
One sentence conclusion.

Content:
{combined_summary}
"""

    final_response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': 'Structured summary generator.'},
            {'role': 'user', 'content': final_prompt}
        ]
    )

    return final_response['message']['content']

def extract_timestamps(transcript_data):

    total_entries = len(transcript_data)

    if total_entries <= 50:
        sampled_data = transcript_data
    else:
        step = max(1, total_entries // 60)
        sampled_data = transcript_data[::step]

    transcript_with_time = ""

    for entry in sampled_data:
        ts = format_timestamp(entry.start)
        transcript_with_time += f"[{ts}] {entry.text}\n"

    prompt = f"""
Select EXACTLY 5 most important moments.

Rules:
• Use ONLY given timestamps
• MM:SS format
• One line per point
• English only

Output EXACTLY:

⏱ Important Timestamps

MM:SS – Description
MM:SS – Description
MM:SS – Description
MM:SS – Description
MM:SS – Description

Transcript:
{transcript_with_time}
"""

    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': 'Strict timestamp selector.'},
            {'role': 'user', 'content': prompt}
        ]
    )

    return response['message']['content']

def normalize_digits(text):
    devanagari_digits = "०१२३४५६७८९"
    english_digits = "0123456789"
    return text.translate(str.maketrans(devanagari_digits, english_digits))


def translate_to_hindi(text):

    prompt = f"""
Translate the following text into natural Hindi.

Rules:
• Hindi ONLY (Devanagari script)
• No English words
• No explanations
• Preserve formatting

Text:
{text}
"""

    try:
        response = ollama.chat(
            model='llama3',
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a professional Hindi translator. Output strictly in Hindi (Devanagari).'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        translated = response['message']['content']

        # ✅ CLEANUP STEP 1 — Remove English letters
        translated = re.sub(r'[A-Za-z]+', '', translated)

        # ✅ CLEANUP STEP 2 — Remove weird/mixed Unicode
        translated = re.sub(r'[^\u0900-\u097F0-9•\n:().\- ]+', ' ', translated)

        translated = normalize_digits(translated)

        # ✅ CLEANUP STEP 3 — Normalize common AI artifacts
        # translated = translated.replace("ट्रंक", "सूंड")
        # translated = translated.replace("ट्रंक्स", "सूंड")

        return translated.strip()

    except Exception as e:
        print("Translation error:", e)
        return "⚠️ Translation error. Showing English version instead."


def translate_to_english(text):
    """
    Translate any non-English transcript into English.
    """

    prompt = f"""
Translate the following text into natural English.

Rules:
• English only
• No explanations
• Preserve meaning accurately

Text:
{text[:4000]}
"""

    try:
        response = ollama.chat(
            model='llama3',
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a professional translator. Output strictly in English.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        return response['message']['content']

    except Exception as e:
        print("Translation error:", e)
        return text  # fallback


def translate_timestamp_block(text):
    lines = text.split("\n")
    result_lines = []

    for line in lines:
        if re.match(r"\d{2}:\d{2}", line.strip()):
            # Split timestamp and description
            parts = line.split("–", 1)
            if len(parts) == 2:
                timestamp = parts[0].strip()
                description = parts[1].strip()

                translated_desc = translate_to_hindi(description)
                result_lines.append(f"{timestamp} – {translated_desc}")
            else:
                result_lines.append(line)
        else:
            # Translate header only
            if "Important Timestamps" in line:
                result_lines.append("⏱ महत्वपूर्ण समय बिंदु")
            else:
                result_lines.append(line)

    return "\n".join(result_lines)

import math

def answer_question(session, question):

    chunks = session["chunks"]
    embeddings = session["embeddings"]

    hindi_question = is_hindi(question)
    lang_rule = "Hindi (Devanagari)" if hindi_question else "English"

    question_embedding = get_embedding(question)

    cleaned_question = clean_text(question)
    question_words = [
        word for word in cleaned_question.split()
        if word not in STOPWORDS and len(word) > 2
    ]

    question_bigrams = [
        f"{question_words[i]} {question_words[i+1]}"
        for i in range(len(question_words) - 1)
    ]

    total_docs = len(chunks)

    # Compute document frequency for TF-IDF
    doc_freq = {}
    for word in set(question_words):
        count = sum(1 for chunk in chunks if word in clean_text(chunk))
        doc_freq[word] = count if count > 0 else 1

    scored_chunks = []

    for index, (chunk, chunk_emb) in enumerate(zip(chunks, embeddings)):

        chunk_clean = clean_text(chunk)

        # ---------- 1️⃣ Semantic Similarity ----------
        similarity = cosine_similarity(question_embedding, chunk_emb)

        # ---------- 2️⃣ TF-IDF Keyword Score ----------
        keyword_score = 0
        for word in question_words:
            tf = chunk_clean.count(word)
            idf = math.log(total_docs / doc_freq[word])
            keyword_score += tf * idf

        # ---------- 3️⃣ Bigram Boost ----------
        bigram_score = 0
        for bigram in question_bigrams:
            if bigram in chunk_clean:
                bigram_score += 3

        # ---------- 4️⃣ Full Phrase Boost ----------
        phrase_score = 5 if cleaned_question in chunk_clean else 0

        # ---------- 5️⃣ Position Boost ----------
        position_boost = index / total_docs

        # ---------- FINAL COMBINED SCORE ----------
        final_score = (
            similarity * 5 +      # semantic weight
            keyword_score * 2 +   # keyword weight
            bigram_score +
            phrase_score +
            position_boost
        )

        scored_chunks.append((final_score, chunk))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    top_chunks = [chunk for score, chunk in scored_chunks[:3] if score > 0]

    if not top_chunks:
        return "This topic is not covered in the video."

    context = "\n\n".join(top_chunks)

    if len(context) > 6000:
        context = context[:6000]

    prompt = f"""
Answer strictly from transcript context.

If missing → reply EXACTLY:
This topic is not covered in the video.

Respond ONLY in {lang_rule}.

Transcript Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': 'Strict QA assistant.'},
            {'role': 'user', 'content': prompt}
        ]
    )

    return response['message']['content']

# ---------------- BUTTONS ---------------- #

def main_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📄 Summary", callback_data="summary")],
        [InlineKeyboardButton("⏱ Timestamps", callback_data="timestamps")],
        [InlineKeyboardButton("💬 Q&A", callback_data="qa")],
        [InlineKeyboardButton("⛔ Stop", callback_data="stop")]
    ])

def language_menu(feature):
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("English", callback_data=f"{feature}_en"),
            InlineKeyboardButton("Hindi", callback_data=f"{feature}_hi")
        ]
    ])

# ---------------- COMMANDS ---------------- #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hello! Send a YouTube link 🎥"
    )

async def process_video(update: Update, user_message: str, chat_id: int):
    video_id = extract_video_id(user_message)

    if not video_id:
        await update.message.reply_text("❌ Invalid YouTube link.")
        return

    # ---------- CACHE CHECK ----------
    if video_id in transcript_cache:
        cached_data = transcript_cache[video_id]

        user_sessions[chat_id] = {
            **cached_data,
            "last_active": time.time()
        }

        await update.message.reply_text("⚡ Loaded from cache.")
        await update.message.reply_text(
            "✅ Transcript loaded. Choose an option:",
            reply_markup=main_menu()
        )
        return


    title = get_video_title(video_id)
    await update.message.reply_text(f"🎥 {title}")

    loading_msg = await update.message.reply_text("⏳ Fetching transcript...")

    try:
        transcript_data = YouTubeTranscriptApi().fetch(video_id)
        original_transcript = " ".join([entry.text for entry in transcript_data])

        # 🔍 Detect language
        detected_lang = detect_transcript_language(original_transcript)

        if detected_lang != "en":
            await update.message.reply_text(
                "🌍 Non-English transcript detected. Processing for best results..."
            )
            translated_transcript = translate_to_english(original_transcript)
        else:
            translated_transcript = original_transcript

        # ✂️ Create chunks for long transcripts
        chunks = chunk_transcript(translated_transcript)

        # Generate embeddings in parallel (non-blocking)
        tasks = [get_embedding_async(chunk) for chunk in chunks]
        chunk_embeddings = await asyncio.gather(*tasks)

        # ---------- SAVE TO CACHE ----------
        if len(transcript_cache) >= MAX_CACHE_SIZE:
            transcript_cache.pop(next(iter(transcript_cache)))

        transcript_cache[video_id] = {
            "original_transcript": original_transcript,
            "transcript": translated_transcript,
            "chunks": chunks,
            "embeddings": chunk_embeddings,
            "timestamps": transcript_data,
            "detected_language": detected_lang
        }

        if len(chunks) > 25:
            await update.message.reply_text(
                "⚠️ Very long video detected. Processing may take slightly longer..."
            )

        # ✅ Remove loading message
        await loading_msg.delete()

        # ---------- LIMIT SESSION COUNT ----------
        if len(user_sessions) >= MAX_SESSIONS:
            user_sessions.pop(next(iter(user_sessions)))

        user_sessions[chat_id] = {
            "original_transcript": original_transcript,
            "transcript": translated_transcript,
            "chunks": chunks, # always English version for AI processing
            "embeddings": chunk_embeddings,
            "timestamps": transcript_data,
            "detected_language": detected_lang,
            "done": set(),
            "last_active": time.time()
        }

        # Smart shortcut detection
        lower_msg = user_message.lower()

        # ---------------- SUMMARY ----------------
        if "summary" in lower_msg:
            if "hindi" in lower_msg:
                result = translate_to_hindi(generate_summary(chunks))
                await update.message.reply_text(result, reply_markup=feature_menu())
                return
            elif "english" in lower_msg:
                result = generate_summary(chunks)
                await update.message.reply_text(result, reply_markup=feature_menu())
                return
            else:
                await update.message.reply_text(
                    "Choose language:",
                    reply_markup=language_menu("summary")
                )
                return

        # ---------------- TIMESTAMPS ----------------
        if "timestamp" in lower_msg:
            if "hindi" in lower_msg:
                result = translate_timestamp_block(
                    extract_timestamps(transcript_data)
                )
                await update.message.reply_text(result, reply_markup=feature_menu())
                return
            elif "english" in lower_msg:
                result = extract_timestamps(transcript_data)
                await update.message.reply_text(result, reply_markup=feature_menu())
                return
            else:
                await update.message.reply_text(
                    "Choose language:",
                    reply_markup=language_menu("timestamps")
                )
                return

        # ---------------- Q&A ----------------
        if "q&a" in lower_msg or "qa" in lower_msg:
            if "hindi" in lower_msg:
                user_sessions[chat_id]["mode"] = "qa_hi"
                await update.message.reply_text("💬 अपना प्रश्न पूछें:")
                return
            elif "english" in lower_msg:
                user_sessions[chat_id]["mode"] = "qa_en"
                await update.message.reply_text("💬 Ask your question:")
                return
            else:
                await update.message.reply_text(
                    "Choose language:",
                    reply_markup=language_menu("qa")
                )
                return

        await update.message.reply_text(
            "✅ Transcript loaded. Choose an option:",
            reply_markup=main_menu()
        )

    except TranscriptsDisabled:
        await loading_msg.delete()
        await update.message.reply_text("❌ Transcripts disabled.")

    except NoTranscriptFound:
        await loading_msg.delete()
        await update.message.reply_text("❌ No transcript available.")

    except Exception as e:
        await loading_msg.delete()
        await update.message.reply_text(f"⚠️ Error: {str(e)}")


def feature_menu():
    keyboard = [
        [InlineKeyboardButton("📄 Summary", callback_data="summary")],
        [InlineKeyboardButton("⏱ Timestamps", callback_data="timestamps")],
        [InlineKeyboardButton("💬 Q&A", callback_data="qa")],
        [InlineKeyboardButton("⛔ Stop", callback_data="stop")]
    ]
    return InlineKeyboardMarkup(keyboard)

# ---------------- CALLBACK HANDLER ---------------- #

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    cleanup_sessions()
    
    query = update.callback_query

    # Always answer callback immediately
    try:
        await query.answer()
    except Exception:
        pass

    chat_id = query.message.chat_id

    # ---------- STOP ----------
    if query.data == "stop":
        user_sessions.pop(chat_id, None)
        try:
            await query.edit_message_text("👍🏻")
        except Exception:
            await query.message.reply_text("👍🏻")
        return

    # ---------- SESSION CHECK ----------
    if chat_id not in user_sessions:
        try:
            await query.edit_message_text("📩 Send YouTube link first 🎥")
        except Exception:
            await query.message.reply_text("📩 Send YouTube link first 🎥")
        return

    session = user_sessions[chat_id]
    session["last_active"] = time.time()

    # ---------- MAIN MENU BUTTON CLICKED ----------
    if query.data in ["summary", "timestamps", "qa"]:
        try:
            await query.edit_message_text(
                "Choose language:",
                reply_markup=language_menu(query.data)
            )
        except Exception:
            await query.message.reply_text(
                "Choose language:",
                reply_markup=language_menu(query.data)
            )
        return

    # ---------- LANGUAGE SELECTED ----------
    try:
        feature, lang = query.data.split("_")
    except ValueError:
        return

    # Remove language buttons instantly + show loading
    try:
        await query.edit_message_text("⏳ Generating...")
    except Exception:
        pass

    # ---------- QA MODE ----------
    if feature == "qa":
        session["mode"] = "qa"
        try:
            await query.edit_message_text("💬 Ask your question:")
        except Exception:
            await query.message.reply_text("💬 Ask your question:")
        return

    # ---------- GENERATE RESULT ----------
    if feature == "summary":
        result = generate_summary(session["chunks"])

    elif feature == "timestamps":
        result = extract_timestamps(session["timestamps"])

    else:
        return

    # ---------- TRANSLATION ----------
    if lang == "hi":
        if feature == "timestamps":
            result = translate_timestamp_block(result)
        else:
            result = translate_to_hindi(result)

    # ---------- FINAL OUTPUT ----------
    try:
        await query.edit_message_text(
            result,
            reply_markup=feature_menu()
        )
    except Exception:
        await query.message.reply_text(
            result,
            reply_markup=feature_menu()
        )

# ---------------- MESSAGE HANDLER ---------------- #

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    cleanup_sessions()
    
    user_message = update.message.text
    chat_id = update.effective_chat.id
    lower_msg = user_message.lower()

    # ---------- LINK DETECTION ----------
    if "youtube.com" in lower_msg or "youtu.be" in lower_msg:
        await process_video(update, user_message, chat_id)
        return

    # ---------- STOP ----------
    if lower_msg == "stop":
        user_sessions.pop(chat_id, None)
        await update.message.reply_text("👍🏻")
        return

    # ---------- SESSION CHECK ----------
    if chat_id not in user_sessions:
        await update.message.reply_text("📩 Send YouTube link first 🎥")
        return

    session = user_sessions[chat_id]
    session["last_active"] = time.time()

    # ---------- ACTIVE QA MODE ----------
    if session.get("mode") in ["qa_en", "qa_hi"]:
        loading_msg = await update.message.reply_text("⏳ Generating...")

        answer = answer_question(session, user_message)

        if session.get("mode") == "qa_hi":
            answer = translate_to_hindi(answer)

        await loading_msg.delete()
        await update.message.reply_text(answer)
        return

    # ---------- SUMMARY ----------
    if "summary" in lower_msg:
        lang = "hi" if "hindi" in lower_msg else "en"

        loading_msg = await update.message.reply_text("⏳ Generating...")

        result = generate_summary(session["chunks"])

        if lang == "hi":
            result = translate_to_hindi(result)

        await loading_msg.delete()
        await update.message.reply_text(result, reply_markup=feature_menu())
        return

    # ---------- TIMESTAMPS ----------
    if "timestamp" in lower_msg:
        lang = "hi" if "hindi" in lower_msg else "en"

        loading_msg = await update.message.reply_text("⏳ Generating...")

        result = extract_timestamps(session["timestamps"])

        if lang == "hi":
            result = translate_timestamp_block(result)

        await loading_msg.delete()
        await update.message.reply_text(result, reply_markup=feature_menu())
        return

    # ---------- Q&A COMMAND TRIGGER ----------
    if "q&a" in lower_msg or "qa" in lower_msg:
        if "hindi" in lower_msg:
            session["mode"] = "qa_hi"
            await update.message.reply_text("💬 अपना प्रश्न पूछें:")
        else:
            session["mode"] = "qa_en"
            await update.message.reply_text("💬 Ask your question:")
        return

    # ---------- FALLBACK ----------
    await update.message.reply_text(
        "Choose an option:",
        reply_markup=main_menu()
    )

# ---------------- GLOBAL ERROR HANDLER ---------------- #

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    print("⚠️ Exception occurred:")
    print(f"Update: {update}")
    print(f"Error: {context.error}")


# ---------------- MAIN ---------------- #

def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.add_error_handler(error_handler)

    print("🚀 Bot running...")
    app.run_polling(drop_pending_updates=True, close_loop=False)

if __name__ == "__main__":
    main()

