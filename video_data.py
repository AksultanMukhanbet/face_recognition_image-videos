# USAGE
# python video_data.py --input videos/aksultan.mp4 --output dataset/aksultan --detector face_detection_model --skip 4

# import the necessary packages
import numpy as np
import argparse
import threading
import time
import cv2
import os

print("Включается камера")
time.sleep(2)
print("Посмотрите камеру, ждите 7-8 секунд")
time.sleep(2)
class VideoRecorder():
	def __init__(self):
		self.open = True
		self.device_index = 0
		self.fps = 6              
		self.fourcc = "MJPG"       
		self.frameSize = (640,480) 
		self.video_filename = "C:/Users/aksul/Documents/python_tasks/liveness_demo/face-recognition/videos/video.avi"
		# input(self.video_filename)
		self.video_cap = cv2.VideoCapture(self.device_index)
		self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
		self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
		self.frame_counts = 1
		self.start_time = time.time()
	def record(self):
		timer_start = time.time()
		timer_current = 0

		while(self.open==True):
			ret, video_frame = self.video_cap.read()
			if (ret==True):
					self.video_out.write(video_frame)
					self.frame_counts += 1
					time.sleep(0.16)
			else:
				break

	def stop(self):
		if self.open==True:
			self.open=False
			self.video_out.release()
			self.video_cap.release()
			cv2.destroyAllWindows()
		else: 
			pass
		
	def start(self):
		video_thread = threading.Thread(target=self.record)
		video_thread.start()
def start_video_recording(filename):		
	global video_thread
	video_thread = VideoRecorder()
	video_thread.start()
	return filename
def stop_video_recording(filename):
	frame_counts = video_thread.frame_counts
	elapsed_time = time.time() - video_thread.start_time
	recorded_fps = frame_counts / elapsed_time
	# print("количество фреймов: " + str(frame_counts))
	print("общее время видео: " + str(elapsed_time))
	print("кадры в секунду: " + str(recorded_fps))
	video_thread.stop() 

if __name__== "__main__":
	filename = ""	
	start_video_recording(filename)  
	time.sleep(7)
	stop_video_recording(filename)
	print("Видео сохранено")
	time.sleep(2)
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16,
	help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0
path = "C:/Users/aksul/Documents/python_tasks/liveness_demo/face-recognition/dataset/aks"

try:
    os.mkdir(path)
except OSError:
    print ("Создать директорию %s не удалось" % path)
else:
    print ("Успешно создана директория %s " % path)
# loop over frames from the video file stream
while True:
	# grab the frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# increment the total number of frames read thus far
	read += 1

	# check to see if we should process this frame
	if read % args["skip"] != 0:
		continue

	# grab the frame dimensions and construct a blob from the frame
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also
		# means our minimum probability test (thus helping filter out
		# weak detections)
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]

			# write the frame to disk
			p = os.sep.join([args["output"],
				"{}.jpg".format(saved)])
			cv2.imwrite(p, face)
			saved += 1
			print("[INFO] saved {} to disk".format(p))

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()