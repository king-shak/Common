from common.util.math import norm, getAngle, getLength, dot, Vector3, Rotation3, RigidTransform3
import numpy as np
import cv2 as cv
import json

axis = np.float32([[0,0,0], [5,0,0], [0,5,0], [0,0,5]]).reshape(-1,3)

def getTranslation(cameraMatrix, distortionCoefficients, objectPoints, imagePoints, range=None):
    # Sort out our range first
    if range is not None:
        objectPoints = objectPoints[range[0] - 1:range[1]]
        imagePoints = imagePoints[range[0] - 1:range[1]]
    
    # Perform pose estimation, obtain the rotation matrix
    retval, rotationVector, translationVector = cv.solvePnP(objectPoints, imagePoints, cameraMatrix, distortionCoefficients)
    rotationMatrix, jacobianMatrix = cv.Rodrigues(rotationVector)

    # Put the results into a RigidTransform3
    translation = Vector3(translationVector[0], translationVector[1], translationVector[2])
    rotation = Rotation3(rotationMatrix)
    rigidTransform = RigidTransform3(translation, rotation)

    # Project the 3D points onto the image plane
    imgpts, jac = cv.projectPoints(axis, rotationVector, translationVector, cameraMatrix, distortionCoefficients)

    return rigidTransform, imgpts

def getAngleToTarget(rvecs):
    angle = np.pi - np.arccos(dot(norm([rvecs[2][0], rvecs[2][2]]), [0, 1]))
    crossProduct = np.cross([0, 0, 1], rvecs[2])
    if (crossProduct[1] < 0):
        angle*=-1
    return angle

class TargetModel():
    def __init__(self, pathToObjPts):
        self.objPts = json.loads(open(pathToObjPts, 'r').readline())['points']
        self.polarPts = np.zeros((len(self.objPts), 2), dtype=np.float32)
        for i in range(len(self.objPts)):
            vector = [self.objPts[i][0], self.objPts[i][1]]
            length = getLength(vector)
            angle = getAngle([1, 0], norm(vector), False)
            self.polarPts[i][0] = length
            self.polarPts[i][1] = angle