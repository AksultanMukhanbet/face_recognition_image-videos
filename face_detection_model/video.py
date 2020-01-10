import cv2
import threading
import time
import os

print("Включается камера")
time.sleep(2)
# print("Покручивайте голову, вас снимает камера!")
# print("Название видеo:")
class VideoRecorder():
	def __init__(self):
		self.open = True
		self.device_index = 0
		self.fps = 6              
		self.fourcc = "MJPG"       
		self.frameSize = (640,480) 
		self.video_filename = "C:/Users/aksul/Documents/python_tasks/liveness_demo/opencv-face-recognition/videos/video.avi"
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