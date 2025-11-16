import argparse
import os
import urllib.request

def _str2bool(v):
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y")

# PURPOSE: Parsing the command line input and extracting the user entered values
# PARAMETERS: N/A
# RETURN:
# - Labels of COCO dataset
# - Path to the weight file
# - Path to configuration file
# - Path to the input video
# - Path to the output video
# - Confidence value
# - Threshold value
def parseCommandLineArguments():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--input", required=True,
		help="path to input video")
	ap.add_argument("-o", "--output", required=True,
		help="path to output video")
	ap.add_argument("-y", "--yolo", required=True,
		help="base path to YOLO directory")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	ap.add_argument("-t", "--threshold", type=float, default=0.3,
		help="threshold when applying non-maxima suppression")
	ap.add_argument("-u", "--use-gpu", type=_str2bool, default=False,
		help="boolean indicating if CUDA GPU should be used")
	ap.add_argument("-s", "--input-size", type=int, default=416,
		help="square input size for YOLO (e.g., 320, 416)")
	ap.add_argument("--skip-frames", type=int, default=0,
		help="number of frames to skip between detections")
	ap.add_argument("--display", type=_str2bool, default=True,
		help="show window output")
	ap.add_argument("--fourcc", type=str, default="MJPG",
		help="fourcc for VideoWriter (e.g., MJPG, XVID, MP4V)")

	args = vars(ap.parse_args())

	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
	configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

	if not os.path.exists(weightsPath):
		print("[INFO] yolov3.weights not found, downloading...")
		os.makedirs(args["yolo"], exist_ok=True)
		try:
			urllib.request.urlretrieve(
				"https://pjreddie.com/media/files/yolov3.weights",
				weightsPath,
			)
		except Exception as e:
			raise RuntimeError("Failed to download yolov3.weights: " + str(e))
	
	inputVideoPath = args["input"]
	outputVideoPath = args["output"]
	confidence = args["confidence"]
	threshold = args["threshold"]
	USE_GPU = args["use_gpu"]
	INPUT_SIZE = args["input_size"]
	SKIP_FRAMES = args["skip_frames"]
	USE_DISPLAY = args["display"]
	FOURCC = args["fourcc"]

	return LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath, confidence, threshold, USE_GPU, INPUT_SIZE, SKIP_FRAMES, USE_DISPLAY, FOURCC
