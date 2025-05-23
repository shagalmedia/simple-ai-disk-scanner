try:
    import numpy
    print(f"NumPy version: {numpy.__version__}")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    import sentence_transformers
    print(f"Sentence-Transformers version: {sentence_transformers.__version__}")
    import pandas
    print(f"Pandas version: {pandas.__version__}")
    import PIL
    print(f"Pillow version: {PIL.__version__}")
    import pytesseract
    print(f"Pytesseract version: {pytesseract.get_tesseract_version()}") # Pytesseract version is often checked this way
except ImportError as e:
    print(f"Error importing one of the key libraries during debug check: {e}")
except AttributeError as e:
    print(f"AttributeError during debug check (possibly with pytesseract version): {e}")

print("--- Debug import check complete. Starting main script imports... ---")

# Main script imports follow
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
import xxhash # For fast file hashing
# Ensure 'xxhash' is installed: pip install xxhash

# Hi there! It's Mark.
# Before running smart disk scanner, ensure you have the necessary dependencies:
# pip install filetype Pillow sentence_transformers transformers torchvision PyPDF2 python-docx pytesseract xxhash
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
MAX_WORKERS = max(1, os.cpu_count()) # Dynamically set based on CPU cores
# CLIP_MAX_TOKENS = 60 # No longer needed, SentenceTransformer handles truncation.
BATCH_COMMIT_SIZE = 50
METADATA_BATCH_SIZE = 20
PDF_MAX_PAGES_TO_PROCESS = 20 # Max pages to read from a PDF

############################
# Global Flags
############################
is_paused = threading.Event()
is_stopped = threading.Event()
models_ready_event = threading.Event()
summarizer_model = None
embedding_model_instance = None # Renamed to avoid conflict with embed_model variable in functions

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
    
    # Add file_hash column if not present
    if "file_hash" not in existing_columns:
        cursor.execute("ALTER TABLE files ADD COLUMN file_hash TEXT")
        conn.commit()

    conn.close()

def init_db():
    ensure_db_schema()

def insert_or_replace_file_record(conn: sqlite3.Connection, file_path: str, file_type: str, size: int,
                                  content: Optional[str], tags: Optional[Dict[str, str]],
                                  embeddings: Optional[bytes], metadata: Optional[Dict],
                                  file_hash: Optional[str]):
    cursor = conn.cursor()

    tags_str = json.dumps(tags) if tags else None
    metadata_str = json.dumps(metadata) if metadata else None

    cursor.execute("""
        INSERT OR REPLACE INTO files (path, type, size, content, tags, embeddings, metadata, file_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (file_path, file_type, size, content, tags_str, embeddings, metadata_str, file_hash))
    # Commit will be handled by the caller (worker)

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
        for i, page in enumerate(reader.pages):
            if i >= PDF_MAX_PAGES_TO_PROCESS:
                break
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

def calculate_file_hash(file_path: str) -> Optional[str]:
    """Calculates the xxhash64 of a file and returns its hex digest."""
    try:
        hasher = xxhash.xxh64()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):  # Read in 64KB chunks
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        logger.warning(f"Hash calculation error: File not found at {file_path}")
        return None
    except PermissionError:
        logger.warning(f"Hash calculation error: Permission denied for file {file_path}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during hash calculation for {file_path}: {e}")
        logger.debug(traceback.format_exc()) # Keep debug for detailed unexpected errors
        return None

############################
# Metadata Extraction with exiftool (If you know better library - just let me know.)
############################
def extract_batch_file_metadata(file_paths: list[str]) -> Dict[str, Optional[Dict]]:
    """
    Extracts metadata for a batch of files using exiftool.
    Returns a dictionary mapping each file path to its metadata dict, or None if extraction failed.
    """
    if not file_paths:
        return {}

    results_map: Dict[str, Optional[Dict]] = {path: None for path in file_paths}
    try:
        # Construct the command: exiftool -j <file1> <file2> ...
        # The -G option is added to ensure SourceFile provides the exact path as given in input.
        # The -api "filter=SourceFile eq '${SourceFile}'" is a potential optimization, but let's test without first.
        cmd = ["exiftool", "-j", "-G"] + file_paths
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if process.returncode != 0:
            # This could happen if exiftool itself crashes or a major issue occurs.
            # Minor errors with individual files are often still return code 0 but with error messages in JSON.
            logger.error(f"Exiftool process returned error code {process.returncode} for batch. Stderr: {process.stderr}")
            # All files in this batch will have None metadata
            return results_map

        try:
            metadata_list = json.loads(process.stdout)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON output from exiftool. Output: {process.stdout[:500]}")
            return results_map # All files get None

        if not isinstance(metadata_list, list):
            logger.error(f"Exiftool output was not a list as expected. Output: {str(metadata_list)[:500]}")
            return results_map

        for metadata_item in metadata_list:
            if not isinstance(metadata_item, dict):
                # logger.warning(f"Skipping non-dict item in exiftool output: {metadata_item}")
                continue # Should not happen with valid exiftool JSON output

            source_file = metadata_item.get("SourceFile")
            if source_file and source_file in results_map:
                # Clean up the metadata item by removing SourceFile as it's now the key
                # Also remove other exiftool process-specific temp fields if any (e.g. ExifToolVersion)
                # For now, just pop SourceFile
                metadata_item.pop("SourceFile", None)
                results_map[source_file] = metadata_item
            # else:
                # logger.warning(f"SourceFile '{source_file}' from exiftool output not in the original request list or already processed.")

    except FileNotFoundError:
        logger.error("Exiftool not found. Please ensure it is installed and in PATH.")
        # All files in this batch will have None metadata
        return {path: None for path in file_paths} # Ensure all requested paths are in the map
    except Exception as e:
        logger.error(f"An unexpected error occurred during exiftool batch processing: {e}")
        logger.error(traceback.format_exc())
        # All files in this batch will have None metadata
        return {path: None for path in file_paths} # Ensure all requested paths are in the map
    
    return results_map

############################
# Embeddings and Analysis, need to be revisited.
############################
# def safe_truncate_text(text: str, max_tokens=CLIP_MAX_TOKENS) -> str: # No longer needed
#     tokens = text.strip().split()
#     if len(tokens) > max_tokens:
#         tokens = tokens[:max_tokens]
#     return " ".join(tokens)

def get_text_embeddings(text: str, embed_model) -> Optional[bytes]:
    if not text.strip(): # Keep this check for empty strings
        return None
    # SentenceTransformer's encode method handles truncation based on model.max_seq_length
    try:
        # The input text here is typically a summary, which is already somewhat short.
        # If it's still too long, embed_model.encode will truncate it.
        emb = embed_model.encode([text], convert_to_tensor=True) 
        return embeddings_to_bytes(emb[0])
    except Exception as e:
        logger.error(f"Error getting text embeddings: {e}")
        # Optionally log traceback.format_exc() if more detail is needed
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

def analyze_file(file_path: str, summarizer, embed_model, metadata: Optional[Dict]) -> Tuple[str, int, str, Optional[bytes], Dict[str, str], Optional[str]]:
    file_hash = calculate_file_hash(file_path)
    
    try:
        size = os.path.getsize(file_path)
    except FileNotFoundError: # Should have been caught by calculate_file_hash, but as a safeguard
        logger.warning(f"File not found when trying to get size for {file_path}, hash was {file_hash}")
        # If hash is None because file not found, this won't be reached due to earlier return in worker (expected).
        # If hash is None for other reasons, we might still want to proceed if size can be obtained.
        # However, if file is not found, we probably shouldn't proceed.
        # For now, if hash is None, we might not even call analyze_file if we check earlier.
        # Let's assume for now if hash is None, we might still try to get other info,
        # but the DB record will reflect the missing hash.
        # A more robust approach might be to return None early from analyze_file if hash is None and file not found.
        # For now, let's proceed and allow size to fail if file is gone.
        size = 0 # Default size if error
    except PermissionError:
        logger.warning(f"Permission error when trying to get size for {file_path}")
        size = 0 # Default size if error


    kind = filetype.guess(file_path) # This can also fail if file is unreadable
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

    # Metadata is now passed as an argument

    if file_type.startswith("text"):
        content, embeddings, tags = analyze_text_file(file_path, summarizer, embed_model)
    elif file_type.startswith("image"):
        content, embeddings, tags = analyze_image_file(file_path, embed_model)
    else:
        # For unknown types, just store minimal info, no embeddings
        content, embeddings, tags = "", None, {"type": file_type or "unknown_file"}

    return file_type, size, content, embeddings, tags, file_hash
# Note: The 'metadata' and 'file_hash' are handled by the worker / analyze_file 
# and passed directly to insert_or_replace_file_record

############################
# Indexing. Eventually indexing should work on the background when you're not using your device.
############################
def index_directory(directory: str, reindex: bool, progress_bar, summarizer, embed_model):
    indexed_files_set = get_indexed_files()
    # errors list is removed, errors will be logged directly

    # 1. Collect all files and filter those that need processing
    all_files_in_dir = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            all_files_in_dir.append(os.path.join(root, file_name))

    files_to_process = []
    if reindex:
        files_to_process = all_files_in_dir
    else:
        for f_path in all_files_in_dir:
            if f_path not in indexed_files_set:
                files_to_process.append(f_path)
    
    if not files_to_process:
        logger.info("No new files to index.")
        progress_bar.close()
        return

    # Update progress bar total to only account for files that will be processed
    progress_bar.total = len(files_to_process)
    progress_bar.refresh()

    # Inner worker function, now takes metadata as an argument
    def worker(file_path: str, metadata: Optional[Dict]):
        conn = None
        # Note: processed_in_worker for DB batching is reset per file path here.
        # This means each file is its own mini-batch for DB commit,
        # unless BATCH_COMMIT_SIZE is 1.
        # This needs to be re-evaluated if BATCH_COMMIT_SIZE is > 1.
        # For now, let's assume BATCH_COMMIT_SIZE=1 for simplicity here,
        # or the DB batching needs to be managed across worker calls if workers are very short-lived.
        # The previous implementation had processed_in_worker inside worker,
        # and worker was long-lived processing multiple files.
        # Given the new structure, we will make each worker commit its own file.
        # Or, more correctly, the DB batching should be per connection, and connections are per worker.
        # Let's restore the original DB batching logic within each worker instance.
        # This means the `worker` function should not be an inner function if we want to maintain
        # its state (like processed_in_worker count for DB batching) across multiple file submissions
        # that might be handled by the *same* worker thread.
        # However, ThreadPoolExecutor reuses threads, but doesn't guarantee a specific thread for a task.
        #
        # Simplest for now: Each worker invocation handles one file and commits it.
        # This means BATCH_COMMIT_SIZE (for DB) effectively becomes 1 for this model.
        # This is a trade-off for simpler batching of metadata.
        # Let's stick to the previous DB batching as it was per worker connection.
        # The worker will process one file, get one connection, do the work, commit, close.
        # This means BATCH_COMMIT_SIZE is effectively 1 from the worker's perspective.
        # This is a consequence of changing worker to process one file from the main loop.

        # Re-evaluating the worker structure for DB batching:
        # The `worker` is defined inside `index_directory` and submitted to the executor.
        # Each call to `worker` will get a file_path and its metadata.
        # The DB connection and batching should still be per worker *thread*.
        # This is complex if the worker function itself doesn't loop.
        #
        # Let's simplify: The `worker` function will process ONE file. It will open a connection,
        # process the file, call insert_or_replace_file_record, commit, and close.
        # The BATCH_COMMIT_SIZE will effectively be 1. This is a performance regression for DB commits
        # but simplifies the current refactoring. We can address DB batching later if needed.

        # CORRECTED WORKER LOGIC (original DB batching was implicitly per thread over multiple files):
        # The previous worker was called with ONE file_path, but it was one of many files handled by the *same*
        # ThreadPoolExecutor worker thread. The connection and processed_in_worker count were local to that
        # worker's execution context for that specific file.
        # The new model: main thread batches for metadata, then submits individual files to executor.
        # Each `worker` call processes one file.
        # So, each worker call *must* handle its own connection and commit. BATCH_COMMIT_SIZE for DB is 1.
        
        conn = None
        try:
            # Analyze file (CPU bound) - now also returns file_hash
            # file_hash might be None if calculation failed.
            file_type, size, content, embeddings, tags, file_hash = analyze_file(file_path, summarizer, embed_model, metadata)
            
            # If file_hash is None and size is 0 due to FileNotFoundError from get_size inside analyze_file,
            # we might want to skip this record.
            # However, analyze_file is designed to return data even if some parts fail.
            # The hash function already logs if file is not found.
            # Let's assume we always try to insert what we have.

            # DB operation
            conn = sqlite3.connect(DB_NAME)
            insert_or_replace_file_record(
                conn,
                file_path=file_path,
                file_type=file_type,
                size=size,
                content=content,
                tags=tags,
                embeddings=embeddings,
                metadata=metadata, # metadata is passed to worker
                file_hash=file_hash # pass file_hash to db record
            )
            conn.commit() # Commit for this single file
        except Exception: # Catching general Exception to log it
            logger.exception(f"Error processing file {file_path} in worker:")
            # No longer appending to errors list
        finally:
            if conn:
                conn.close()
        return file_path


    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {}
        for i in range(0, len(files_to_process), METADATA_BATCH_SIZE):
            if is_stopped.is_set():
                logger.info("Stopping metadata batching and further processing due to stop signal.")
                break
            
            current_batch_paths = files_to_process[i:i + METADATA_BATCH_SIZE]
            
            # Log the batch being sent to exiftool
            # logger.info(f"Extracting metadata for batch of {len(current_batch_paths)} files...")
            batch_metadata_map = extract_batch_file_metadata(current_batch_paths)
            # logger.info(f"Received metadata for {sum(1 for md in batch_metadata_map.values() if md is not None)} files in batch.")

            for file_path_in_batch in current_batch_paths:
                if is_stopped.is_set():
                    break
                
                # Handle pause inside the loop before submitting to executor
                while is_paused.is_set():
                    time.sleep(0.5)
                if is_stopped.is_set(): # Check again after pause
                    break

                current_file_metadata = batch_metadata_map.get(file_path_in_batch)
                # Submit to executor for analysis and DB insertion
                future = executor.submit(worker, file_path_in_batch, current_file_metadata)
                future_to_path[future] = file_path_in_batch
            
            if is_stopped.is_set(): # After submitting a batch's files
                 logger.info("Stop signal received, breaking from submitting more file batches.")
                 break
        
        # Process results as they complete
        for future in as_completed(future_to_path):
            path_processed = future_to_path[future]
            try:
                result = future.result() # To catch exceptions from worker if any (already caught in worker though)
                # if result:
                #    logger.info(f"Successfully processed: {path_processed}")
            except Exception as e:
                # This should ideally be caught and logged within the worker itself.
                # errors.append(f"Error processing {path_processed} from future: {e}")
                # errors.append(traceback.format_exc())
                pass # Already handled in worker
            progress_bar.update(1)

    progress_bar.close()

    # Fetch total count from DB after all workers are done
    # This connection is separate and short-lived, which is fine.
    final_conn = sqlite3.connect(DB_NAME)
    final_cursor = final_conn.cursor()
    final_cursor.execute("SELECT COUNT(*) FROM files")
    total_indexed = final_cursor.fetchone()[0]
    final_conn.close()

    logger.info(f"Indexing finished. Total indexed files in the database: {total_indexed}")
    logger.info(f"Database path: {DB_NAME}")

    # The errors list and its logging loop are removed. Errors are logged in real-time.

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
def load_models_background():
    """Loads models in the background and sets the models_ready_event."""
    global summarizer_model, embedding_model_instance
    try:
        logger.info("Starting background model loading...")
        summarizer_model = pipeline("summarization", model="philschmid/bart-large-cnn-samsum", from_pt=True)
        logger.info("Summarization model loaded.")
        
        embedding_model_instance = SentenceTransformer("clip-ViT-B-32")
        embedding_model_instance.max_seq_length = 77 # Ensure this is set for the global model
        logger.info("Embedding model loaded.")
        
        models_ready_event.set()
        logger.info("All models loaded and ready.")
    except Exception as e:
        logger.error(f"Fatal error during background model loading: {e}")
        logger.error(traceback.format_exc())
        # If models fail to load, the app might be in an unusable state for indexing.
        # Consider how to handle this - perhaps set an error flag or exit.
        # For now, the event won't be set, and indexing will hang or fail.

def main():
    global is_paused, is_stopped, summarizer_model, embedding_model_instance
    init_db()

    # Start loading models in a background thread
    model_loader_thread = threading.Thread(target=load_models_background, daemon=True)
    model_loader_thread.start()

    logger.info("Welcome to the Smart Disk Scanner!")
    logger.info("You can index files from selected directory. This program will:")
    logger.info("- Extract text and summarize it for text files.")
    logger.info("- Perform OCR on images, store embeddings and basic tags.")
    logger.info("- Extract metadata from files and store it.")
    logger.info("- Store embeddings for semantic search (future functionality).")
    logger.info("Commands: index <folder_path>, pause, resume, stop, exit")

    index_thread = None

    # Models are now global: summarizer_model and embedding_model_instance

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
                # Check if models are loaded, wait if not
                if not models_ready_event.is_set():
                    logger.info("Models are still loading, please wait...")
                    models_ready_event.wait() # Wait for models to be ready
                    logger.info("Models are now ready. Proceeding with indexing.")
                
                if summarizer_model is None or embedding_model_instance is None:
                    logger.error("Models could not be loaded. Cannot start indexing. Please check logs.")
                    continue

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

                # Use the globally loaded models
                current_summarizer = summarizer_model
                current_embed_model = embedding_model_instance

                def run_index():
                    try:
                        index_directory(directory, reindex, progress_bar, current_summarizer, current_embed_model)
                    except Exception:
                        logger.exception("An unexpected error occurred during the main indexing process in run_index:")

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
