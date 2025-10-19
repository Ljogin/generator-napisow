import os
import tempfile
import warnings
from pathlib import Path

import streamlit as st
from pydub import AudioSegment
import openai

# ──────────────────────────────────────────────────────────────────────────────
# 0) Konfiguracja kluczy i ffmpeg
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Aplikacja Generowanie Napisów", page_icon="🎬", layout="centered")

# Klucz API z Streamlit Secrets, .env lub środowiska
OPENAI_API_KEY = None
try:
    if "OPENAI_API_KEY" in st.secrets:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

if not OPENAI_API_KEY:
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Brak OPENAI_API_KEY. Dodaj go w Streamlit Secrets lub w pliku .env")
    st.stop()

openai.api_key = OPENAI_API_KEY

# Opcjonalny katalog ffmpeg
FFMPEG_DIR = None
try:
    FFMPEG_DIR = st.secrets.get("FFMPEG_DIR", None)
except Exception:
    pass

if FFMPEG_DIR:
    os.environ["PATH"] += os.pathsep + FFMPEG_DIR
    os.environ["FFMPEG_BINARY"] = str(Path(FFMPEG_DIR) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg"))
    os.environ["FFPROBE_BINARY"] = str(Path(FFMPEG_DIR) / ("ffprobe.exe" if os.name == "nt" else "ffprobe"))

warnings.filterwarnings("ignore", message="Couldn't find ffmpeg")
warnings.filterwarnings("ignore", message="Couldn't find ffprobe")

# Inicjalizacja stanu aplikacji
if "step" not in st.session_state:
    st.session_state["step"] = "upload"

# ──────────────────────────────────────────────────────────────────────────────
# 1) UI + opis kroków
# ──────────────────────────────────────────────────────────────────────────────
st.title("🎬 Aplikacja Generowanie Napisów")
st.caption("v1–v5: upload wideo → ekstrakcja audio → transkrypcja (Whisper) → pobranie")

with st.expander("Plan / Taski (specyfikacja)", expanded=False):
    st.markdown(
        """
- **v1** – użytkownik może przesłać plik wideo i my go wyświetlamy  
- **v2** – wyodrębniamy dźwięk z wideo i również go wyświetlamy  
- **v3** – wykorzystujemy model speech-to-text w celu wygenerowania napisów  
- **v4** – po transkrypcji można pobrać napisy  
- **v5** – aplikacja wraca do kroku 2 po zakończeniu
        """
    )

# ──────────────────────────────────────────────────────────────────────────────
# 2) Funkcje pomocnicze
# ──────────────────────────────────────────────────────────────────────────────
def save_uploaded_file_to_temp(uploaded_file) -> Path:
    """Zapisuje UploadedFile do pliku tymczasowego z odpowiednim rozszerzeniem."""
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def extract_audio_to_mp3(video_path: Path) -> Path:
    """Wyodrębnia audio z wideo do MP3 przy użyciu pydub (ffmpeg)."""
    audio_seg = AudioSegment.from_file(str(video_path))
    out_path = Path(tempfile.mkstemp(suffix=".mp3")[1])
    audio_seg.export(str(out_path), format="mp3")
    return out_path


def transcribe_audio(audio_path: Path, response_format: str = "srt") -> str:
    """Transkrybuje audio z użyciem modelu Whisper-1."""
    with open(audio_path, "rb") as f:
        transcription = openai.Audio.transcribe("whisper-1", f)
    return transcription["text"] if isinstance(transcription, dict) else str(transcription)


def bytes_for_download(text: str) -> bytes:
    return text.encode("utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# 3) Główna logika aplikacji
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("1) Prześlij wideo")
uploaded = st.file_uploader(
    "Obsługiwane: mp4, mov, mkv, webm, avi",
    type=["mp4", "mov", "mkv", "webm", "avi"],
)

if uploaded:
    st.video(uploaded)
    video_tmp = save_uploaded_file_to_temp(uploaded)

    st.divider()
    st.subheader("2) Wyodrębnij audio z wideo")
    col1, col2 = st.columns(2)
    with col1:
        do_extract = st.button("🔊 Wyodrębnij audio (MP3)", type="primary")
    with col2:
        resp_format = st.selectbox("Format napisów:", ["srt", "text"], index=0)

    if do_extract:
        try:
            audio_mp3_path = extract_audio_to_mp3(video_tmp)
            st.audio(str(audio_mp3_path), format="audio/mp3")
            st.success("Audio wyodrębnione ✔️")

            st.divider()
            st.subheader("3) Generuj napisy (Whisper-1)")

            if st.button("🧠 Transkrybuj audio", type="primary"):
                with st.spinner("Transkrypcja w toku…"):
                    captions = transcribe_audio(audio_mp3_path)
                st.success("Transkrypcja zakończona ✔️")

                # ✅ Zamiast edycji i dalszych kroków — tylko pobranie + powrót
                st.download_button(
                    "⬇️ Pobierz napisy",
                    bytes_for_download(captions),
                    file_name="captions.srt" if resp_format == "srt" else "captions.txt",
                    mime="text/plain",
                )

                st.info("✅ Napisy zostały wygenerowane. Możesz wrócić i przetworzyć kolejne wideo.")

                if st.button("⬅️ Powrót do kroku 2 (Wyodrębnij audio)"):
                    st.session_state["step"] = "extract_audio"
                    st.experimental_rerun()

        except Exception as e:
            st.error(f"Wystąpił błąd: {e}")
else:
    st.info("Załaduj plik wideo, aby rozpocząć.")