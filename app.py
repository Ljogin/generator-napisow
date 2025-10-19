import os
import io
import tempfile
import warnings
from pathlib import Path

import streamlit as st
from pydub import AudioSegment
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# 0) Konfiguracja kluczy i ffmpeg (działa w Streamlit Cloud oraz lokalnie)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Aplikacja Generowanie Napisów", page_icon="🎬", layout="centered")

# Sekrety: najpierw Streamlit, potem .env, potem zmienne środowiskowe
OPENAI_API_KEY = None
try:
    if "OPENAI_API_KEY" in st.secrets:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

if not OPENAI_API_KEY:
    # awaryjnie: spróbuj z .env lub systemu
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Brak OPENAI_API_KEY. Dodaj go w Streamlit Secrets lub w pliku .env")
    st.stop()

# Opcjonalnie: podaj ścieżkę do ffmpeg/ffprobe przez Secrets (np. na komputerze firmowym)
FFMPEG_DIR = None
try:
    FFMPEG_DIR = st.secrets.get("FFMPEG_DIR", None)  # np. "C:\\ffmpeg\\bin"
except Exception:
    pass

if FFMPEG_DIR:
    os.environ["PATH"] += os.pathsep + FFMPEG_DIR
    os.environ["FFMPEG_BINARY"] = str(Path(FFMPEG_DIR) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg"))
    os.environ["FFPROBE_BINARY"] = str(Path(FFMPEG_DIR) / ("ffprobe.exe" if os.name == "nt" else "ffprobe"))

# Usuń ostrzeżenia pydub o ffmpeg
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg")
warnings.filterwarnings("ignore", message="Couldn't find ffprobe")

client = OpenAI(api_key=OPENAI_API_KEY)


# ──────────────────────────────────────────────────────────────────────────────
# 1) UI + opis kroków
# ──────────────────────────────────────────────────────────────────────────────
st.title("🎬 Aplikacja Generowanie Napisów")
st.caption("v1–v5: upload wideo → ekstrakcja audio → transkrypcja (Whisper) → edycja → pobranie")

with st.expander("Plan / Taski (specyfikacja)", expanded=False):
    st.markdown(
        """
- **v1** – użytkownik może przesłać plik wideo i my go wyświetlamy  
- **v2** – wyodrębniamy dźwięk z wideo i również go wyświetlamy  
- **v3** – wykorzystujemy model speech-to-text w celu wygenerowania napisów i je wyświetlamy  
- **v4** – napisy mogą być edytowane  
- **v5** – poprawione napisy można pobrać jako plik  
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
    """
    Wyodrębnia audio z wideo do MP3 używając pydub (ffmpeg).
    Zapisuje do pliku tymczasowego i zwraca jego ścieżkę.
    """
    # pydub potrzebuje ffmpeg/ffprobe w PATH
    audio_seg = AudioSegment.from_file(str(video_path))
    out_path = Path(tempfile.mkstemp(suffix=".mp3")[1])
    audio_seg.export(str(out_path), format="mp3")
    return out_path


def transcribe_audio(audio_path: Path, response_format: str = "srt") -> str:
    """
    Transkrybuje audio z użyciem modelu Whisper-1.
    response_format: "srt" (napisy z czasem) lub "text" (ciągły tekst)
    """
    # OpenAI SDK v1 – audio.transcriptions.create
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format=response_format,  # "srt" lub "text"
            temperature=0.0,
        )

    # Zwracany typ to zwykle str dla "srt"/"text"
    return transcription if isinstance(transcription, str) else str(transcription)


def bytes_for_download(text: str) -> bytes:
    return text.encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# 3) Główna logika aplikacji
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("1) Prześlij wideo")
uploaded = st.file_uploader(
    "Obsługiwane: mp4, mov, mkv, webm, avi",
    type=["mp4", "mov", "mkv", "webm", "avi"],
    accept_multiple_files=False,
)

if uploaded:
    # v1 – pokaż wideo
    st.video(uploaded)

    # Zapisz do pliku tymczasowego (PIL/pydub lub ffmpeg lepiej pracują z plikami)
    video_tmp = save_uploaded_file_to_temp(uploaded)

    st.divider()
    st.subheader("2) Wyodrębnij audio z wideo")
    col1, col2 = st.columns(2)
    with col1:
        do_extract = st.button("🔊 Wyodrębnij audio (MP3)", type="primary")
    with col2:
        resp_format = st.selectbox("Format napisów do wygenerowania (krok 3):", ["srt", "text"], index=0)

    if do_extract:
        try:
            audio_mp3_path = extract_audio_to_mp3(video_tmp)
            # v2 – pokaż audio
            st.audio(str(audio_mp3_path), format="audio/mp3")
            st.success("Audio wyodrębnione ✔️")

            st.divider()
            st.subheader("3) Generuj napisy (Whisper-1)")

            if st.button("🧠 Transkrybuj audio do napisów", type="primary"):
                with st.spinner("Transkrypcja w toku…"):
                    captions = transcribe_audio(audio_mp3_path, response_format=resp_format)
                st.success("Transkrypcja zakończona ✔️")

                # v3 + v4 – pokaż edytowalne napisy
                st.subheader("4) Edytuj napisy")
                help_text = "Możesz poprawić napisy poniżej. Jeśli wybrałeś 'srt' – zachowaj format znaczników czasu."
                edited = st.text_area("Napisy", value=captions, height=300, help=help_text)

                st.subheader("5) Pobierz napisy")
                default_name = "captions.srt" if resp_format == "srt" else "captions.txt"
                st.download_button(
                    label="⬇️ Pobierz napisy",
                    data=bytes_for_download(edited),
                    file_name=default_name,
                    mime="text/plain",
                )

                # Dodatkowy eksport czystego txt (opcjonalnie)
                if resp_format == "srt":
                    st.download_button(
                        label="⬇️ Pobierz napisy jako .txt (bez czasu)",
                        data=bytes_for_download(edited),
                        file_name="captions.txt",
                        mime="text/plain",
                        key="dl_txt_plain",
                    )

        except Exception as e:
            st.error(f"Wystąpił błąd: {e}")
            st.info(
                "Jeśli to błąd ffmpeg/ffprobe, upewnij się, że:\n"
                "- w Streamlit Cloud jest zainstalowany pakiet systemowy `ffmpeg` (patrz packages.txt),\n"
                "- lokalnie masz ffmpeg w PATH lub podaj FFMPEG_DIR w Secrets."
            )
else:
    st.info("Załaduj plik wideo, aby rozpocząć.")