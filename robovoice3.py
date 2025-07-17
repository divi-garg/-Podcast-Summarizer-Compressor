import os
import json
from pytube import extract
from openai import OpenAI
from gtts import gTTS
from pydub import AudioSegment
import whisper
import yt_dlp

# ---------- Configuration ----------
client = OpenAI(api_key="YOUR_KEY")  # Replace with your actual key
WHISPER_MODEL = "small"  # Options: tiny, base, small, medium, large

# ---------- Step 1: Download YouTube Audio ----------
def get_video_id(youtube_url):
    return extract.video_id(youtube_url)

def download_youtube_audio(youtube_url, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)
    video_id = extract.video_id(youtube_url)
    output_path = os.path.join(output_dir, f"{video_id}.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
        audio_file = os.path.join(output_dir, f"{video_id}.mp3")

    return audio_file, video_id

# ---------- Step 2: Transcribe Full Audio ----------
def transcribe_audio(audio_path, video_id):
    print("🔍 Transcribing full audio...")
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path)
    transcript_path = f"{video_id}_transcript.json"
    with open(transcript_path, "w") as f:
        json.dump(result, f, indent=2)
    return result['text']

# ---------- Step 3: Summarize Transcript Using GPT ----------
def summarize_transcript(full_text, target_parts=3):
    print("🧠 Generating multi-part summary...")
    word_count = len(full_text.split())
    chunk_size = word_count // target_parts

    chunks = [full_text[i:i + chunk_size] for i in range(0, word_count, chunk_size)]
    full_summary = ""

    for i, chunk in enumerate(chunks):
        print(f"🔹 Summarizing part {i+1}/{len(chunks)}...")
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional podcast editor. Your task is to create a 15–20 minute podcast summary script. Use a natural tone with storytelling flow."
                },
                {
                    "role": "user",
                    "content": f"Generate part {i+1} of a full 15–20 minute detailed podcast summary. This part should sound natural and continuous. Focus on a detailed summary of the following segment:\n\n{chunk}"
                }
            ],
            temperature=0.5
        )
        full_summary += response.choices[0].message.content.strip() + "\n\n"

    return full_summary

# ---------- Step 4: Convert Text to Full Robotized Audio ----------
def generate_robot_voice(summary_text, output_path):
    print("🎤 Generating full-length robot voice using gTTS...")

    chunk_size = 4000  # gTTS safe limit
    chunks = [summary_text[i:i + chunk_size] for i in range(0, len(summary_text), chunk_size)]

    final_audio = AudioSegment.empty()

    for i, chunk in enumerate(chunks):
        temp_path = f"temp_chunk_{i}.mp3"
        try:
            tts = gTTS(text=chunk)
            tts.save(temp_path)
            final_audio += AudioSegment.from_mp3(temp_path)
        except Exception as e:
            print(f"⚠️ Failed to generate chunk {i}: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    final_audio.export(output_path, format="mp3")
    print(f"✅ Full summary audio saved to: {output_path}")

# ---------- Master Function ----------
def podcast_summary_pipeline(youtube_url):
    print(f"🚀 Processing: {youtube_url}")
    audio_path, video_id = download_youtube_audio(youtube_url)
    transcript = transcribe_audio(audio_path, video_id)
    summary = summarize_transcript(transcript)
    print("\n🔑 Summary:\n", summary[:500] + "\n...")  # Print just preview
    output_path = f"{video_id}_robot_summary.mp3"
    generate_robot_voice(summary, output_path)

# ---------- Run ----------
if __name__ == "__main__":
    podcast_summary_pipeline("https://youtu.be/2SRVN9f25v4")

