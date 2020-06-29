import numpy as np
import cv2 as cv
import json

class USBCamera:
    def __init__(self, cameraConfigFilePath, port=0):
        cameraConfig = open(cameraConfigFilePath, 'r')
        self.resolution = json.loads(cameraConfig.readline())
        self.framerate = json.loads(cameraConfig.readline())
        self.cameraMatrix = np.array(json.loads(cameraConfig.readline()))
        self.distortionCoefficients = np.array(json.loads(cameraConfig.readline()))
        self.frameMidpoint = json.loads(cameraConfig.readline())
        self.frameSize = json.loads(cameraConfig.readline())
        cameraConfig.close()

        self.port = port
        self.cap = cv.VideoCapture(port)
        if not self.cap.isOpened():
            raise ValueError('Unable to access camera at port ', port)

        # self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        # self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

    def getFrame(self):
        if self.cap.isOpened():
            return self.cap.read()
        else:
            raise ValueError('Unable to access camera at port ', self.port)
    
    def getResolution(self):
        return self.resolution

    def getFramerate(self):
        return self.framerate

    def getCameraMatrix(self):
        return self.cameraMatrix

    def getDistortionCoefficients(self):
        return self.distortionCoefficients

    def getFrameMidpoint(self):
        return self.frameMidpoint

    def getFrameSize(self):
        return self.frameSize

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

class CameraConfig:
    def __init__(self, cameraConfigFilePath):
        cameraConfig = open(cameraConfigFilePath, 'r')
        self.resolution = json.loads(cameraConfig.readline())
        self.framerate = json.loads(cameraConfig.readline())
        self.cameraMatrix = np.array(json.loads(cameraConfig.readline()))
        self.distortionCoefficients = np.array(json.loads(cameraConfig.readline()))
        self.frameMidpoint = json.loads(cameraConfig.readline())
        self.frameSize = json.loads(cameraConfig.readline())

    def getResolution(self):
        return self.resolution

    def getFramerate(self):
        return self.framerate

    def getCameraMatrix(self):
        return self.cameraMatrix

    def getDistortionCoefficients(self):
        return self.distortionCoefficients

    def getFrameMidpoint(self):
        return self.frameMidpoint

    def getFrameSize(self):
        return self.frameSize