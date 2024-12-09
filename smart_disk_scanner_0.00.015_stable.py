import os
import sys
import sqlite3
import threading
import time
import signal
import logging
import traceback
import warnings
import json
from typing import Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# Hi there! It's Mark.
# Before running smart disk scanner, ensure you have the necessary dependencies:
# pip install filetype Pillow sentence_transformers transformers torchvision PyPDF2 python-docx pytesseract
# Additionally:
# - Ensure 'tesseract' is installed (for OCR). On macOS: brew install tesseract
# - For metadata extraction, we will rely on 'exiftool' CLI. Please install exiftool:
#   macOS: brew install exiftool
#   Linux: apt-get install libimage-exiftool-perl (or similar)
#   Windows: Download from exiftool website.
#
# exiftool is a very robust tool that can extract metadata from a wide range of file formats
# (images, videos, documents, audio, etc.). We call it via subprocess and parse the JSON output.
#
# Key features of this app:
# - Index files recursively in a given directory.
# - Extract text from text files, summarize them, store embeddings (not working perfetly, yet).
# - Perform OCR on images, store embeddings, extract metadata (unstable).
# - Extract metadata from any file using exiftool, store it in the database (JSON).
# - Store embeddings for semantic search (future functionality, to enable local files search by the context).
# - Pause/resume/stop indexing.
# - Avoid dimension mismatch in CLIP model by truncating text aggressively, not sure how to solve issues related to it, yet.
#
# Best practices:
# - Check and create DB schema if needed.
# - If new columns are required, add them.
# - Catch errors and log them.
# - Efficient indexing with ThreadPoolExecutor, if you know better way - improve it then.
#
# Please ensure exiftool is installed and accessible in PATH.
# If exiftool is not installed, metadata extraction will fail gracefully and store no metadata.
# Also tested with MD5 generation for every file, but its too slow, should i try bcrypt? or else, if you have ideas, let me know.

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

import filetype
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from tqdm import tqdm
from PyPDF2 import PdfReader
from docx import Document
from transformers import pipeline, logging as transformers_logging

transformers_logging.set_verbosity_error()
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

############################
# Config
############################
DB_FOLDER = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DB_FOLDER, exist_ok=True)
DB_NAME = os.path.join(DB_FOLDER, "file_index.db")

MAX_TEXT_LENGTH = 50_000
SUMMARY_MAX_LENGTH = 100
MAX_WORKERS = 4
CLIP_MAX_TOKENS = 60  # Aggressive truncation to avoid dimension mismatch (!)

############################
# Global Flags
############################
is_paused = threading.Event()
is_stopped = threading.Event()

############################
# Logging
############################
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

############################
# Database Functions. Sorry for any inconveniece, not an expert in databases. Thinking of Mongo or PostgreSQL.
############################
def ensure_db_schema():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Base table. 
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS files (
        path TEXT PRIMARY KEY,
        type TEXT,
        size INTEGER,
        content TEXT,
        tags TEXT,
        embeddings BLOB
    )
    """)
    conn.commit()

    # Check existing columns
    cursor.execute("PRAGMA table_info(files)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Add metadata column if not present (In case of any issues with db you could just delete the db file if there is nothing important, also please backup your db before experiments)
    if "metadata" not in existing_columns:
        cursor.execute("ALTER TABLE files ADD COLUMN metadata TEXT")
        conn.commit()

    conn.close()

def init_db():
    ensure_db_schema()

def insert_or_replace_file_record(file_path: str, file_type: str, size: int,
                                  content: Optional[str], tags: Optional[Dict[str, str]],
                                  embeddings: Optional[bytes], metadata: Optional[Dict]):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    tags_str = json.dumps(tags) if tags else None
    metadata_str = json.dumps(metadata) if metadata else None

    cursor.execute("""
        INSERT OR REPLACE INTO files (path, type, size, content, tags, embeddings, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (file_path, file_type, size, content, tags_str, embeddings, metadata_str))
    conn.commit()
    conn.close()

def get_indexed_files() -> set:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT path FROM files")
    indexed = {row[0] for row in cursor.fetchall()}
    conn.close()
    return indexed

############################
# File Reading. Need to be improved seriously.
############################
def read_text_file(file_path: str, max_length=MAX_TEXT_LENGTH) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_length)
    except Exception:
        return ""

def read_pdf_file(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        texts = []
        for page in reader.pages[:20]:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)[:MAX_TEXT_LENGTH]
    except Exception:
        return ""

def read_docx_file(file_path: str) -> str:
    try:
        doc = Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        return "\n".join(full_text)[:MAX_TEXT_LENGTH]
    except Exception:
        return ""

def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".txt", ".md", ".csv", ".html"]:
        return read_text_file(file_path)
    elif ext == ".pdf":
        return read_pdf_file(file_path)
    elif ext == ".docx":
        return read_docx_file(file_path)
    else:
        # Default fallback
        return read_text_file(file_path)

############################
# Helper Functions
############################
def embeddings_to_bytes(embedding_tensor) -> bytes:
    arr = embedding_tensor.cpu().numpy().tolist()
    import struct
    return struct.pack(f"{len(arr)}f", *arr)

############################
# Metadata Extraction with exiftool (If you know better library - just let me know.)
############################
def extract_file_metadata(file_path: str) -> Optional[Dict]:
    # We call exiftool to extract metadata in JSON format (Have idea to store EXIF data separately but don't know best practices)
    # exiftool -j file_path
    # If exiftool is not installed or fails, return None, lmao
    try:
        result = subprocess.run(["exiftool", "-j", file_path], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        if isinstance(data, list) and len(data) > 0:
            # exiftool returns a list of dicts, usually one element
            # Remove some unnecessary keys if needed.
            # But, i mean, we can store as is.
            # Potentially remove "SourceFile" key as it's redundant, i dunno, don't have time for it.
            md = data[0]
            md.pop("SourceFile", None)
            return md
        return None
    except Exception:
        return None

############################
# Embeddings and Analysis, need to be revisited.
############################
def safe_truncate_text(text: str, max_tokens=CLIP_MAX_TOKENS) -> str:
    tokens = text.strip().split()
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return " ".join(tokens)

def get_text_embeddings(text: str, embed_model) -> Optional[bytes]:
    if not text.strip():
        return None
    truncated_text = safe_truncate_text(text, CLIP_MAX_TOKENS)
    try:
        emb = embed_model.encode([truncated_text], convert_to_tensor=True)
        return embeddings_to_bytes(emb[0])
    except Exception:
        return None

def summarize_text(content: str, summarizer) -> str:
    if len(content.split()) < 40:
        return content.strip()
    try:
        summary = summarizer(content, max_length=SUMMARY_MAX_LENGTH, min_length=20, do_sample=False)[0]["summary_text"]
        return summary.strip()
    except:
        return content.strip()

def ocr_image(image: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(image, lang="eng")
        return text.strip()
    except Exception:
        return ""

def analyze_text_file(file_path: str, summarizer, embed_model) -> Tuple[str, Optional[bytes], Dict[str, str]]:
    content = extract_text_from_file(file_path)
    if not content.strip():
        return "", None, {}
    summary = summarize_text(content, summarizer)
    embeddings = get_text_embeddings(summary, embed_model)
    tags = {
        "type": "text_file",
        "length": str(len(content)),
        "original_excerpt": content[:5000]
    }
    return summary, embeddings, tags

def analyze_image_file(file_path: str, embed_model) -> Tuple[str, Optional[bytes], Dict[str, str]]:
    try:
        image = Image.open(file_path).convert("RGB")
    except Exception:
        return "", None, {"type": "image_file", "error": "cannot_open"}

    ocr_text = ocr_image(image)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    emb_bytes = None
    try:
        emb = embed_model.encode(image_tensor, convert_to_tensor=True)
        emb_bytes = embeddings_to_bytes(emb[0])
    except:
        emb_bytes = None

    tags = {
        "type": "image_file",
        "ocr_length": str(len(ocr_text)),
        "has_ocr": "true" if ocr_text else "false"
    }

    content = ocr_text[:2000] if ocr_text else "Image file (no OCR text extracted)"
    return content, emb_bytes, tags

def analyze_file(file_path: str, summarizer, embed_model) -> Tuple[str, int, str, Optional[bytes], Dict[str, str], Optional[Dict]]:
    size = os.path.getsize(file_path)
    kind = filetype.guess(file_path)
    if kind:
        file_type = kind.mime
    else:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".heic"]:
            file_type = "image/" + ext.replace(".", "")
        elif ext in [".txt", ".md", ".csv", ".html", ".pdf", ".docx"]:
            file_type = "text/plain"
        else:
            file_type = "unknown"

    metadata = extract_file_metadata(file_path)

    if file_type.startswith("text"):
        content, embeddings, tags = analyze_text_file(file_path, summarizer, embed_model)
    elif file_type.startswith("image"):
        content, embeddings, tags = analyze_image_file(file_path, embed_model)
    else:
        # For unknown types, just store minimal info, no embeddings
        content, embeddings, tags = "", None, {"type": file_type or "unknown_file"}

    return file_type, size, content, embeddings, tags, metadata

############################
# Indexing. Eventually indexing should work on the background when you're not using your device.
############################
def index_directory(directory: str, reindex: bool, progress_bar, summarizer, embed_model):
    indexed_files = get_indexed_files()

    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))

    errors = []

    def worker(file_path):
        while is_paused.is_set():
            time.sleep(0.5)
        if is_stopped.is_set():
            return None
        if not reindex and file_path in indexed_files:
            return None
        try:
            file_type, size, content, embeddings, tags, metadata = analyze_file(file_path, summarizer, embed_model)
            insert_or_replace_file_record(
                file_path=file_path,
                file_type=file_type,
                size=size,
                content=content,
                tags=tags,
                embeddings=embeddings,
                metadata=metadata
            )
        except Exception as e:
            err_msg = f"Error indexing {file_path}: {e}"
            errors.append(err_msg)
            traceback_str = ''.join(traceback.format_exc())
            errors.append(traceback_str)
        return file_path

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker, f) for f in all_files]
        for fut in as_completed(futures):
            if is_stopped.is_set():
                break
            progress_bar.update(1)

    progress_bar.close()

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM files")
    total_indexed = cursor.fetchone()[0]
    conn.close()

    logger.info(f"Indexing finished. Total indexed files in the database: {total_indexed}")
    logger.info(f"Database path: {DB_NAME}")

    if errors:
        logger.info("Some errors occurred during indexing:")
        for e in errors:
            logger.info(e)

############################
# Signal Handler
############################
def signal_handler(sig, frame):
    global is_stopped
    logger.info("Received interrupt signal. Stopping indexing...")
    is_stopped.set()

signal.signal(signal.SIGINT, signal_handler)

############################
# Main!
############################
def main():
    global is_paused, is_stopped
    init_db()

    logger.info("Welcome to the Smart Disk Scanner!")
    logger.info("You can index files from selected directory. This program will:")
    logger.info("- Extract text and summarize it for text files.")
    logger.info("- Perform OCR on images, store embeddings and basic tags.")
    logger.info("- Extract metadata from files and store it.")
    logger.info("- Store embeddings for semantic search (future functionality).")
    logger.info("Commands: index <folder_path>, pause, resume, stop, exit")

    index_thread = None

    summarizer = None
    embed_model = None

    while True:
        action = input("\nEnter a command: ").strip().lower()
        if action.startswith("index"):
            parts = action.split(" ", 1)
            if len(parts) < 2:
                logger.info("Usage: index <folder_path>")
                continue
            directory = parts[1]
            if not os.path.exists(directory):
                logger.info(f"The directory '{directory}' does not exist.")
                continue
            reindex = input("Reindex existing files? It will override exisitng data (yes/no): ").strip().lower() == "yes"

            if index_thread and index_thread.is_alive():
                logger.info("Indexing is already in progress. Stop it first before starting a new one.")
            else:
                # Load models here once we know we are indexing
                if summarizer is None:
                    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum", from_pt=True)
                if embed_model is None:
                    embed_model = SentenceTransformer("clip-ViT-B-32")
                    embed_model.max_seq_length = 77

                is_paused.clear()
                is_stopped.clear()

                total_files = 0
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        total_files += 1
                if total_files == 0:
                    logger.info("Sorry, no files found in the given directory.")
                    continue

                progress_bar = tqdm(total=total_files, desc="Indexing progress", unit="file")

                def run_index():
                    try:
                        index_directory(directory, reindex, progress_bar, summarizer, embed_model)
                    except Exception as e:
                        logger.error(f"Unexpected error during indexing: {e}")
                        traceback.print_exc()

                index_thread = threading.Thread(target=run_index, daemon=True)
                index_thread.start()

        elif action == "pause":
            if index_thread and index_thread.is_alive():
                is_paused.set()
                logger.info("Indexing paused. Use 'resume' to continue.")
            else:
                logger.info("No indexing in progress to pause.")

        elif action == "resume":
            if index_thread and index_thread.is_alive():
                is_paused.clear()
                logger.info("Indexing resumed.")
            else:
                logger.info("No indexing in progress to resume.")

        elif action == "stop":
            if index_thread and index_thread.is_alive():
                is_stopped.set()
                index_thread.join()
                logger.info("Indexing stopped.")
            else:
                logger.info("No indexing in progress to stop.")

        elif action == "exit":
            if index_thread and index_thread.is_alive():
                logger.info("Stopping indexing before exit...")
                is_stopped.set()
                index_thread.join()
            logger.info("Goodbye, don't forget to hit star on GitHub!")
            break

        else:
            logger.info("Hmm, unknown command. Use commands like: index, pause, resume, stop, exit.")

if __name__ == '__main__':
    main()
