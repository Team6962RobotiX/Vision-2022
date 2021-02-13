# https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/
from picamera import PiCamera
from picamera.array import PiRGBArray

class PiCapture:
	def __init__(self):
		self.cam = PiCamera()
		self.rawCapture = PiRGBArray(self.cam)

	def set(self, var1, var2):
		pass

	def start(self):
		self.cam.start_preview()

	def update(self):
		pass

	def read(self):
		self.cam.capture(self.rawCapture, format='bgr')
		return self.rawCapture.array

	def stop(self):
		self.cam.stop_preview()

	def release(self):
		self.stop()

	def __exit__(self, exec_type, exec_value, traceback):
		self.release()

# TODO: the rest of this