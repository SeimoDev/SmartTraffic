# import the necessary packages
import numpy as np
import time
from scipy import spatial
import cv2
from input_retrieval import *
import os
from concurrent.futures import ThreadPoolExecutor

#All these classes will be counted as 'vehicles'
list_of_vehicles = ["bicycle","car","motorbike","bus","truck", "train"]
# Setting the threshold for the number of frames to search a vehicle for
FRAMES_BEFORE_CURRENT = 10  

#Parse command line arguments and extract the values required
LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath,\
	preDefinedConfidence, preDefinedThreshold, USE_GPU, INPUT_SIZE, SKIP_FRAMES, USE_DISPLAY, FOURCC = parseCommandLineArguments()
cv2.setUseOptimized(True)

def _get_cfg_input_size(cfg_path):
    w, h = None, None
    try:
        with open(cfg_path, "r") as f:
            for line in f:
                if line.startswith("width="):
                    w = int(line.split("=")[-1].strip())
                elif line.startswith("height="):
                    h = int(line.split("=")[-1].strip())
                if w is not None and h is not None:
                    break
    except Exception:
        pass
    return w, h

cfg_w, cfg_h = _get_cfg_input_size(configPath)
if cfg_w and cfg_h:
    if INPUT_SIZE % 32 != 0 or INPUT_SIZE != cfg_w or INPUT_SIZE != cfg_h:
        INPUT_SIZE = cfg_w

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
# PURPOSE: Displays the vehicle count on the top-left corner of the frame
# PARAMETERS: Frame on which the count is displayed, the count number of vehicles 
# RETURN: N/A
def displayVehicleCount(frame, vehicle_count):
	cv2.putText(
		frame, #Image
		'Detected Vehicles: ' + str(vehicle_count), #Label
		(20, 20), #Position
		cv2.FONT_HERSHEY_SIMPLEX, #Font
		0.8, #Size
		(0, 0xFF, 0), #Color
		2, #Thickness
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)

# PURPOSE: Determining if the box-mid point cross the line or are within the range of 5 units
# from the line
# PARAMETERS: X Mid-Point of the box, Y mid-point of the box, Coordinates of the line 
# RETURN: 
# - True if the midpoint of the box overlaps with the line within a threshold of 5 units 
# - False if the midpoint of the box lies outside the line and threshold
def boxAndLineOverlap(x_mid_point, y_mid_point, line_coordinates):
	x1_line, y1_line, x2_line, y2_line = line_coordinates #Unpacking

	if (x_mid_point >= x1_line and x_mid_point <= x2_line+5) and\
		(y_mid_point >= y1_line and y_mid_point <= y2_line+5):
		return True
	return False

# PURPOSE: Displaying the FPS of the detected video
# PARAMETERS: Start time of the frame, number of frames within the same second
# RETURN: New start time, new number of frames 
def displayFPS(start_time, num_frames):
	current_time = int(time.time())
	if(current_time > start_time):
		
		print("FPS:", num_frames)
		num_frames = 0
		start_time = current_time
	return start_time, num_frames

# PURPOSE: Draw all the detection boxes with a green dot at the center
# RETURN: N/A
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			#Draw a green dot in the middle of the box
			cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 0xFF, 0), thickness=2)

# PURPOSE: Initializing the video writer with the output video path and the same number
# of fps, width and height as the source video 
# PARAMETERS: Width of the source video, Height of the source video, the video stream
# RETURN: The initialized video writer
def initializeVideoWriter(video_width, video_height, videoStream):
	# Getting the fps of the source video
	sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
	# initialize our video writer
	os.makedirs(os.path.dirname(outputVideoPath) or ".", exist_ok=True)
	ext = os.path.splitext(outputVideoPath)[1].lower()
	chosen = FOURCC
	if ext == ".mp4":
		chosen = "mp4v"
	fourcc = cv2.VideoWriter_fourcc(*chosen)
	return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
		(video_width, video_height), True)

# PURPOSE: Identifying if the current box was present in the previous frames
# PARAMETERS: All the vehicular detections of the previous frames, 
#			the coordinates of the box of previous detections
# RETURN: True if the box was current box was present in the previous frames;
#		  False if the box was not present in the previous frames
def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
	centerX, centerY, width, height = current_box
	dist = np.inf #Initializing the minimum distance
	# Iterating through all the k-dimensional trees
	for i in range(FRAMES_BEFORE_CURRENT):
		coordinate_list = list(previous_frame_detections[i].keys())
		if len(coordinate_list) == 0: # When there are no detections in the previous frame
			continue
		# Finding the distance to the closest point and the index
		temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
		if (temp_dist < dist):
			dist = temp_dist
			frame_num = i
			coord = coordinate_list[index[0]]

	if (dist > (max(width, height)/2)):
		return False

	# Keeping the vehicle ID constant
	current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
	return True

def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
	current_detections = {}
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			
			centerX = x + (w//2)
			centerY = y+ (h//2)

			# When the detection is in the list of vehicles, AND
			# it crosses the line AND
			# the ID of the detection is not present in the vehicles
			if (LABELS[classIDs[i]] in list_of_vehicles):
				current_detections[(centerX, centerY)] = vehicle_count 
				if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):
					vehicle_count += 1
					# vehicle_crossed_line_flag += True
				# else: #ID assigning
					#Add the current detection mid-point of box to the list of detected items
				# Get the ID corresponding to the current detection

				ID = current_detections.get((centerX, centerY))
				# If there are two detections having the same ID due to being too close, 
				# then assign a new ID to current detection.
				if (list(current_detections.values()).count(ID) > 1):
					current_detections[(centerX, centerY)] = vehicle_count
					vehicle_count += 1 

				#Display the ID at the center of the box
				cv2.putText(frame, str(ID), (centerX, centerY),\
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)

	return vehicle_count, current_detections

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

def _enable_cuda_if_available(net):
    info = cv2.getBuildInformation()
    has_cuda_build = ("NVIDIA CUDA: YES" in info) or ("CUDA: YES" in info)
    has_cuda_device = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    if has_cuda_build and has_cuda_device:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            return True
        except Exception:
            pass
    return False

def _enable_opencl_if_available(net):
    try:
        if cv2.ocl.haveOpenCL():
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            return True
    except Exception:
        pass
    return False

if USE_GPU:
    if not _enable_cuda_if_available(net):
        print("[WARN] CUDA not available. Using CPU backend.")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
else:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

ln = net.getLayerNames()
unconnected = net.getUnconnectedOutLayers()
try:
    ln = [ln[i - 1] for i in unconnected.flatten()]
except Exception:
    ln = [ln[int(i) - 1] for i in unconnected]

# initialize the video stream, pointer to output video file, and
# frame dimensions
videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Specifying coordinates for a default line 
x1_line = 0
y1_line = video_height//2
x2_line = video_width
y2_line = video_height//2

#Initialization
previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
# previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
num_frames, vehicle_count = 0, 0
writer = initializeVideoWriter(video_width, video_height, videoStream)
start_time = int(time.time())
prev_boxes, prev_confidences, prev_classIDs, prev_idxs = [], [], [], []
executor = ThreadPoolExecutor(max_workers=1)
detection_future = None
# loop over frames from the video file stream
while True:
	print("================NEW FRAME================")
	num_frames+= 1
	print("FRAME:\t", num_frames)
	# Initialization for each iteration
	boxes, confidences, classIDs = [], [], [] 
	vehicle_crossed_line_flag = False 

	#Calculating fps each second
	start_time, num_frames = displayFPS(start_time, num_frames)
	# read the next frame from the file
	(grabbed, frame) = videoStream.read()

	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break

	# construct a blob from the input frame and then perform detection asynchronously
	do_detect = (SKIP_FRAMES <= 0) or (num_frames % (SKIP_FRAMES + 1) == 0)
	if do_detect:
		if detection_future is None or detection_future.done():
			def _detect_async(frame_copy, vw, vh):
				blob = cv2.dnn.blobFromImage(frame_copy, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
				net.setInput(blob)
				outputs = net.forward(ln)
				boxes_, confidences_, classIDs_ = [], [], []
				for output in outputs:
					for detection in output:
						scores = detection[5:]
						classID = int(np.argmax(scores))
						confidence = float(scores[classID])
						if confidence > preDefinedConfidence:
							box = detection[0:4] * np.array([vw, vh, vw, vh])
							(centerX, centerY, width, height) = box.astype("int")
							x = int(centerX - (width / 2))
							y = int(centerY - (height / 2))
							boxes_.append([x, y, int(width), int(height)])
							confidences_.append(confidence)
							classIDs_.append(classID)
				idxs_ = cv2.dnn.NMSBoxes(boxes_, confidences_, preDefinedConfidence, preDefinedThreshold)
				return boxes_, confidences_, classIDs_, idxs_
			detection_future = executor.submit(_detect_async, frame.copy(), video_width, video_height)

	if detection_future is not None and detection_future.done():
		prev_boxes, prev_confidences, prev_classIDs, prev_idxs = detection_future.result()

	boxes, confidences, classIDs, idxs = prev_boxes, prev_confidences, prev_classIDs, prev_idxs

	# # Changing line color to green if a vehicle in the frame has crossed the line 
	# if vehicle_crossed_line_flag:
	# 	cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0xFF, 0), 2)
	# # Changing line color to red if a vehicle in the frame has not crossed the line 
	# else:
	# 	cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0, 0xFF), 2)

	# apply non-maxima suppression handled in async detection; reuse previous results

	# Draw detection box 
	drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

	vehicle_count, current_detections = count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame)

	# Display Vehicle Count if a vehicle has passed the line 
	displayVehicleCount(frame, vehicle_count)

	# write the output frame to disk
	writer.write(frame)

	if USE_DISPLAY:
		cv2.imshow('Frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break	
	
	# Updating with the current frame detections
	previous_frame_detections.pop(0) #Removing the first frame from the list
	# previous_frame_detections.append(spatial.KDTree(current_detections))
	previous_frame_detections.append(current_detections)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
videoStream.release()
try:
	executor.shutdown(wait=False)
except Exception:
	pass
