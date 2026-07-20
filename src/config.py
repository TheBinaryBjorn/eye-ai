"""Central place for tunable constants shared across modules."""

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
BOX_COLOR = (0, 0, 255)  # BGR red

WINDOW_NAME = "Object Recognition"
PANEL_WIDTH = 420
PANEL_BG_COLOR = (30, 30, 30)
PANEL_HEADING_COLOR = (0, 0, 255)
PANEL_TEXT_COLOR = (230, 230, 230)
PANEL_SUBTEXT_COLOR = (160, 160, 160)

QUIT_KEY = "q"
VOICE_KEY = "v"  # press to start recording a question, press again to stop and send it

# Voice pipeline: microphone -> transcription -> llm -> synthesis -> speaker
MIC_SAMPLE_RATE = 16000
WHISPER_MODEL_SIZE = "base.en"
LLM_MODEL = "llama3.2"
OLLAMA_HOST = "http://localhost:11434"

# A Piper voice is two files (an .onnx model and its .onnx.json config).
# Download one from https://github.com/rhasspy/piper/blob/master/VOICES.md
# and point these at the files, e.g. via `python -m piper.download_voices`.
PIPER_MODEL_PATH = "models/en_US-lessac-medium.onnx"
PIPER_CONFIG_PATH = "models/en_US-lessac-medium.onnx.json"
