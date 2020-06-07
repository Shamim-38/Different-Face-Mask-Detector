# USAGE
# python detect_mask_video.py
#for_webcam
#python3 detect_mask_video.py --model checkpoint/model-054-0.978078-0.951172.h5 --output output/video/5.mp4
#for video
# python3 detect_mask_video.py --video test/mask.mp4 --model checkpoint/model-054-0.978078-0.951172.h5 --output output/video/5.mp4


# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 180x180, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (180, 180))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		preds = maskNet.predict(faces)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to face detector model directory")	
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="checkpoint/mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output", type=str,
	default="output.mp4",
	help="path to output detector")	
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(1)

"""
print(type(args["video"]))

if args["video"] is not False:
	# Open the video file
	if not os.path.isfile(args["video"]):
		print("Input video file ", args.video, " doesn't exist")
		sys.exit(1)
	vs = cv2.VideoCapture(args["video"])
	#out_file_path = str(out_file_path / (args.video[:-4] + '_tf_out.mp4'))
else:
	# Webcam input
	vs = cv2.VideoCapture(0)
	# 设置摄像头像素值
	#out_file_path = str(out_file_path / 'webcam_tf_out.mp4')

"""

#vs = cv2.VideoCapture("v5.mkv")
#time.sleep(2.0)
writer = None
(W, H) = (None, None)

# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")

 
#out = vs.Writer('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
 


# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 224x224, and then
	# perform mean subtraction
	output = frame.copy()
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#frame = cv2.resize(frame, (224, 224)).astype("float32")
	#frame -= mean
	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(with_N95_mask, with_dust_mask, with_surgical_mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		if (with_N95_mask > with_dust_mask) and (with_N95_mask > with_surgical_mask) and (with_N95_mask > withoutMask):
			label = "Mask_N95"
			color = (255, 128, 0)
		elif (with_dust_mask > with_N95_mask) and (with_dust_mask > with_surgical_mask) and (with_dust_mask > withoutMask):
			label = "Mask_Dust"
			color = (255, 153, 255)
		elif (with_surgical_mask > with_N95_mask) and (with_surgical_mask > with_dust_mask) and (with_surgical_mask > withoutMask):
			label = "Mask_Surgical"
			color = (255,255,0)            
		else:
			label = "No Mask"
			color = (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(with_dust_mask, with_N95_mask, withoutMask, with_surgical_mask) * 100)


		# display the label and bounding box rectangle on the output
		# frame
		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)

		# write the output frame to disk
		writer.write(frame)		 
        
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)

		# write the output frame to disk
		writer.write(frame)	
		
    
	# show the output frame
	cv2.imshow("Frame", frame)
		# check if the video writer is None


    
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
#vs.stop()
