from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO,]

# Images config
IMAGES_DIR = ROOT / 'images/sample_images'
DEFAULT_IMAGE = 'images/DJI_0023.jpg'
DEFAULT_DETECT_IMAGE = 'images/DJI_0023_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'data/sample_videos'
VIDEO_1_PATH = VIDEO_DIR / 'Video_1p5s.mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
}

# ML Model config
MODEL_DIR = ROOT / 'models'
DETECTION_MODEL = MODEL_DIR / 'vision_giant.onnx'
SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0
