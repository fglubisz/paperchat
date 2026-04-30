"""paperchat — chat with technical PDFs using RAG."""
import logging
import os
import warnings

# Quiet down noisy library output — must run before any HF/transformers imports
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Suppress warnings emitted via the warnings module
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# Suppress logger output from huggingface_hub (the "unauthenticated requests" message
# is emitted via logging, not warnings — needs a different mute)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)