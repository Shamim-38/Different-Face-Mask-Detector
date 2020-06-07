# USAGE
# python detect_mask_image.py --image examples/example_01.png
#python3 detect_mask_image.py --image examples/example_05.jpeg --face face_detector --model checkpoint/model-054-0.978078-0.951172.h5 --output output/images/2.png


# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="checkpoint/mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output", type=str,
	default="output_03.jpg",
	help="path to output detector")		
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model(args["model"])

# load the input image from disk, clone it, and grab the image spatial
# dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# pass the blob through the network and obtain the face detections
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

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
		# ordering, resize it to 224x224, and preprocess it
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (180, 180))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)
        
		# pass the face through the model to determine if the face
		# has a mask or not
		(with_N95_mask, with_dust_mask, with_surgical_mask, withoutMask) = model.predict(face)[0]
         
        
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
			label = "Without_Mask"
			color = (0, 0, 255)
			
        


		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(with_dust_mask, with_N95_mask, withoutMask, with_surgical_mask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
             
# show the output image
cv2.imshow("Output", image)
cv2.imwrite(args["output"], image)
cv2.waitKey(0)