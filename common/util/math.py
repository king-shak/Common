from math import cos, sin, atan2, sqrt, pi
from scipy.spatial import distance as dist
import numpy as np
import cv2 as cv
    
horizontal = [1., 0.]

def getPrincipalAxes(contourPoints):
    mean = np.empty((0))
    mean, eigenvectors, _ = cv.PCACompute2(contourPoints, mean)
    x = [eigenvectors[0][0], eigenvectors[1][0]]
    y = [eigenvectors[0][1], eigenvectors[1][1]]
    rotation = getAngle(horizontal, x, False)
    return x, y, np.ravel(mean), rotation

# Legacy code
def rotatePoint(pt, angle):
    x = (pt[0] * np.cos(angle)) - (pt[1] * np.sin(angle))
    y = (pt[1] * np.cos(angle)) + (pt[0] * np.sin(angle))
    return [x, y]

# TO-DO: Remove this and use np.dot
def dot(a, b):     
    return (a[0] * b[0]) + (a[1] * b[1])

# TO-DO: refactor when getLength is deprecated
def norm(vector):
    return vector / getLength(vector)

# TO-DO: Remove this and just use np.linalg.norm()
def getLength(vector):
    return np.linalg.norm(vector)

def getRelativeAngleDirection(a, b):
    return ((a[0] * b[1]) - (a[1] * b[0])) > 0

def getAngle(a, b, signedRange = None):
    rotation = np.arccos(round(dot(a, b), 6) / round((getLength(a) * getLength(b)), 6))
    if signedRange is not None:
        sign = getRelativeAngleDirection(a, b)
        if (not sign):
            if (signedRange):
                rotation = rotation * -1.0
            else :
                rotation = (2 * np.pi) - rotation
    return rotation

# Rotates and expands an image to avoid cropping
def rotateImage(img, angle):
    # Get the dimensions and center of the image
    height, width = img.shape[:2]
    imgCenter = (width / 2, height / 2)
    
    # Now get our ratation matrix
    rotationMatrix = cv.getRotationMatrix2D(imgCenter, angle, 1)

    # Take the absolute value of the cos and sin from the rotation matrix
    absoluteCos = abs(rotationMatrix[0, 0])
    absoluteSin = abs(rotationMatrix[0, 1])

    # Find the new width and height bounds
    widthBound = int(height * absoluteSin + width * absoluteCos)
    heightBound = int(height * absoluteCos + width * absoluteSin)

    # Subtract the old image center from the rotation matrix (essentially beringing it back to the origin) and add the new corrdinates
    rotationMatrix[0, 2] += widthBound / 2 - imgCenter[0]
    rotationMatrix[1, 2] += heightBound / 2 - imgCenter[1]

    # Finally rotate the image with our modified rotation matrix
    rotatedImg = cv.warpAffine(img, rotationMatrix, (widthBound, heightBound))
    return rotatedImg

# Line Class
class Line:
	def __init__(self, directionVector, point):
		self.directionVector = directionVector
		self.point = point
		self.a = directionVector[0]
		self.b = directionVector[1]
		self.xSubNought = self.point[0]
		self.ySubNought = self.point[1]
		
	def getPoint(self, t, rounded):
		x = self.xSubNought + (self.a * t)
		y = self.ySubNought - (self.b * t)
		if rounded:
			x = int(round(x))
			y = int(round(y))
		return (x, y)
	
	# returns the point at which this line intersects another
	def intersects(self, other):
		a = np.array([
			[self.a, -1 * other.a],
			[self.b, -1 * other.b]
			], dtype=np.float32)
        
		c = np.array([
			[other.xSubNought + (-1 * self.xSubNought)],
			[other.ySubNought + (-1 * self.ySubNought)],
			], dtype=np.float32)
        
		intersects = True

		try:
			a_inv = np.linalg.inv(a)
		except:
			print('these two lines do not intersect!')
			intersects = False

		if intersects:
			result = np.matmul(a_inv, c)
			
			# now we calculate the point at which it intersects given t and s
			x = round(self.xSubNought + self.a * result[0][0])
			y = round(self.ySubNought + self.b * result[0][0])

			return [x, y]

# Rotation3
class Rotation3:
    def __init__(self, rotationMatrix):
        self.rotationMatrix = rotationMatrix
        self.eulerAngles = Rotation3.rotationMatrixToEulerAngles(rotationMatrix)

    def inverse(self):
        return Rotation3(np.transpose(self.rotationMatrix))

    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6
 
    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(R) :
        assert(Rotation3.isRotationMatrix(R))
        sy = sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = atan2(R[2, 1], R[2, 2])
            y = atan2(-R[2, 0], sy)
            z = atan2(R[1, 0], R[0, 0])
        else:
            x = atan2(-R[1, 2], R[1, 1])
            y = atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])

    def __add__(self, other):
        sum = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                sum[i][j] = self.rotationMatrix[i][j] + other.rotationMatrix[i][j]
        return Rotation3(sum)

    def __str__(self):
        return 'Yaw:' + str(round(self.eulerAngles[0], 2)) + ', Pitch:' + str(round(self.eulerAngles[1], 2)) + ', Roll:' + str(round(self.eulerAngles[2], 2))


# Vector3
class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

        self.length = np.linalg.norm([x, y, z])

    def scale(self, scale):
        return Vector3(self.x * scale, self.y * scale, self.z * scale)

    def negate(self):
        return Vector3(self.x * -1.0, self.y * -1.0, self.z * -1.0)

    def norm(self):
        return Vector3(self.x / self.length, self.y / self.length, self.z / self.length)

    def dot(self, other):
        return np.dot([self.x, self.y, self.z], [other.x, other.y, other.z])

    def cross(self, other):
        x, y, z = np.cross([self.x, self.y, self.z], [other.x, other.y, other.z])
        return Vector3(x, y, z)

    def rotate(self, rotation):
        result_x = self.x * rotation.rotationMatrix[0][0] + self.y * rotation.rotationMatrix[1][0] + self.z * rotation.rotationMatrix[2][0]
        result_y = self.x * rotation.rotationMatrix[0][1] + self.y * rotation.rotationMatrix[1][1] + self.z * rotation.rotationMatrix[2][1]
        result_z = self.x * rotation.rotationMatrix[0][2] + self.y * rotation.rotationMatrix[1][2] + self.z * rotation.rotationMatrix[2][2]
        return Vector3(result_x, result_y, result_z)

    def __mul__(self, other):
        return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __str__(self):
        return 'X:' + str(self.x) + ', Y:' + str(self.y) + ', Z:' + str(self.z)


# RigidTransform3
class RigidTransform3:
    def __init__(self, translation, rotation):
        self.translation = translation
        self.rotation = rotation

    def inverse(self, other):
        return RigidTransform3(self.translation.negate(), self.rotation.inverse())
    
    def __add__(self, other):
        return RigidTransform3(self.translation + other.translation, self.rotation + other.rotation)

    def __str__(self):
        return str(self.translation) + '\n' + str(self.rotation)