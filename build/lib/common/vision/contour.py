from common.util.math import norm, getAngle, getLength, horizontal, Line
from scipy.spatial import distance as dist
import numpy as np
import collections
import cv2 as cv

# Some math helper methods to find the properties of the contours
def getMidPoint(pts):
    moment = cv.moments(pts)
    midPtX = int(moment['m10'] / moment['m00'])
    midPtY = int(moment['m01'] / moment['m00'])
    return (midPtX, midPtY)

def sortImgPts(imgpts, x, midpt):
    numOfPoints = len(imgpts)
    pts = {}
    for i in range(numOfPoints):
        vector = norm([imgpts[i][0][0] - midpt[0], midpt[1] - imgpts[i][0][1]])
        angle = getAngle(x, vector, False)
        pts[angle] = imgpts[i]
    
    pts = collections.OrderedDict(sorted(pts.items()))
    
    sortedPts = np.zeros((numOfPoints, 1, 2), dtype=np.int32)
    j = 0
    for i in pts:
        sortedPts[j] = pts[i]
        j+=1

    return sortedPts

def sortRectPoints(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def getBoundingBoxPoints(points):
    x, y, w, h = cv.boundingRect(points)
    boundingBoxPoints = np.array([[x, y],
                                [x + w, y],
                                [x + w, y - h],
                                [x, y - h]
                                ], dtype=np.float32)
    # return the bounding box verticies, the area of the bounding box, and the aspect ratio of the width and height of the boudning box
    return np.array(boundingBoxPoints, dtype=np.uint8), w * h, w / h

def getReferenceVector(points):
    points = sortRectPoints(points)

    lowestPointVal = points[0][1]
    lowestPointIndex = 0
    for i in range(len(points)):
        if (points[i][1] > lowestPointVal):
            lowestPointVal = points[i][1]
            lowestPointIndex = i
        
    points = np.concatenate((points[lowestPointIndex:], points[:lowestPointIndex]))

    # vector a
    a = [points[1][0] - points[0][0], points[0][1] - points[1][1]]
    width = getLength(a)

    # vector b
    b = [points[2][0] - points[1][0], points[1][1] - points[2][1]]
    height = getLength(b)

    # vector d
    d = [points[3][0] - points[0][0], points[0][1] - points[3][1]]
    d = norm(d)

    angle = np.degrees(getAngle(horizontal, d))

    if (width > height):
        angle = 270 + angle

    return [np.cos(np.radians(angle)), np.sin(np.radians(angle))]

class Contour:
    def __init__(self, contourPoints, frameCenter, useConvexHull, numOfCorners):
        # First process the contour points
        self.points = contourPoints

        # Find the area of the contour
        self.area = cv.contourArea(self.points)

        # Obtain the straight bounding box
        self.boundingBoxPoints, self.boundingBoxArea, self.boundingBoxAspectRatio = getBoundingBoxPoints(self.points)
        x, y, self.boundingBoxWidth, self.boundingBoxHeight = cv.boundingRect(self.points)
        self.boundingBoxArea = self.boundingBoxHeight * self.boundingBoxWidth
        self.boundingBoxAspectRatio = self.boundingBoxWidth / self.boundingBoxHeight
        self.boundingBoxUpperLeftPoint = (x, y)
        self.boundingBoxLowerRightPoint = (x + self.boundingBoxWidth, y + self.boundingBoxHeight)

        # Find the rotated rect and it's area
        rect = cv.minAreaRect(self.points)
        _, (width, height), _ = rect
        if (width < height):
            self.tshort = width
            self.tlong = height
        else:
            self.tshort = height
            self.tlong = width
        box = np.int0(cv.boxPoints(rect))
        self.rotatedRect = [box]
        self.rotatedRectArea = cv.contourArea(box)

        # Compute the verticies of the contour
        self.vertices = cv.approxPolyDP(self.points, 0.015 * cv.arcLength(self.points, True), True)

        # Apply k-means if there are duplicates
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        if (len(self.vertices) > numOfCorners) and (numOfCorners is not 0):
                self.vertices = cv.kmeans(self.vertices.astype(dtype=np.float32), numOfCorners, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)[2]
                self.vertices = self.vertices.reshape((numOfCorners, 1, 2)).astype(dtype=np.int32)

        # Find the convex hull of the contour
        self.useConvexHull = useConvexHull
        self.convexHull = cv.convexHull(self.vertices)

        # Find the midpoint of the contour
        self.midpoint = getMidPoint(self.points)

        # Find the direction vector using the rotated rect and create a Line instance of it
        self.directionVector = getReferenceVector(box)
        self.rotation = getAngle(horizontal, self.directionVector, True)
        self.referenceVector = Line(self.directionVector, self.midpoint)
        self.contourLine = Line(self.directionVector, [self.midpoint[0], frameCenter[1] * 2 - self.midpoint[1]])

        # Finally, sort the vertices
        if self.useConvexHull:
            self.vertices = sortImgPts(self.convexHull, self.directionVector, self.midpoint).astype(dtype=np.int32)
        else:
            self.vertices = sortImgPts(self.vertices, self.directionVector, self.midpoint).astype(dtype=np.int32)
        
        # Get the distance to the center of the frame (used for sorting)
        self.distanceToCenter = np.linalg.norm(np.array([self.midpoint[0] - frameCenter[0], frameCenter[1] - self.midpoint[1]]))

class ContourGroup:
    def __init__(self, contours, frameCenter, useConvexHull, numOfCorners):
        # Save the contours that comprise the group
        self.contours = contours

        # Combine the points from the contours
        self.vertices = ContourGroup.combinePoints(contours)

        # Find the convex hull of the contour group
        self.useConvexHull = useConvexHull
        self.convexHull = cv.convexHull(self.vertices)

        # Apply k-means if there are duplicates
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        if useConvexHull:
            if (len(self.convexHull) > numOfCorners) and (numOfCorners is not 0):
                self.convexHull = cv.kmeans(self.convexHull.astype(dtype=np.float32), numOfCorners, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)[2]
                self.convexHull = self.convexHull.reshape((numOfCorners, 1, 2)).astype(dtype=np.int32)
        else:
            if (len(self.vertices) > numOfCorners) and (numOfCorners is not 0):
                self.vertices = cv.kmeans(self.vertices.astype(dtype=np.float32), numOfCorners, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)[2]
                self.vertices = self.vertices.reshape((numOfCorners, 1, 2)).astype(dtype=np.int32)

        # Find the area of the contour
        self.area = ContourGroup.combineArea(contours)

        # Obtain the straight bounding box
        self.boundingBoxPoints, self.boundingBoxArea, self.boundingBoxAspectRatio = getBoundingBoxPoints(self.vertices)
        x, y, self.boundingBoxWidth, self.boundingBoxHeight = cv.boundingRect(self.vertices)
        self.boundingBoxUpperLeftPoint = (x, y)
        self.boundingBoxLowerRightPoint = (x + self.boundingBoxWidth, y + self.boundingBoxHeight)
        self.boundingBoxPoints = [self.boundingBoxUpperLeftPoint, [x + self.boundingBoxWidth, y], self.boundingBoxLowerRightPoint, [x, y + self.boundingBoxHeight]]

        # Find the rotated rect and it's area
        rect = cv.minAreaRect(self.vertices)
        _, (width, height), _ = rect
        if (width < height):
            self.tshort = width
            self.tlong = height
        else:
            self.tshort = height
            self.tlong = width
        box = np.int0(cv.boxPoints(rect))
        self.rotatedRect = [box]
        self.rotatedRectArea = cv.contourArea(box)
        
        # Get the center of the group
        self.midpoint = tuple(np.average(self.vertices, axis=0).ravel().astype(int))

        # Find the direction vector using the rotated rect and create a Line instance of it
        self.directionVector = getReferenceVector(box)
        self.rotation = getAngle(horizontal, self.directionVector)
        self.referenceVector = Line(self.directionVector, self.midpoint)
        self.contourLine = Line(self.directionVector, [self.midpoint[0], frameCenter[1] * 2 - self.midpoint[1]])

        # Finally, sort the image points
        self.vertices = sortImgPts(self.vertices, self.directionVector, self.midpoint)

        # Get the distance to the center of the frame
        self.distanceToCenter = np.linalg.norm(np.array([self.midpoint[0] - frameCenter[0], frameCenter[1] - self.midpoint[1]]))
    
    def combinePoints(contours):
        pts = np.concatenate((contours[0].vertices, contours[1].vertices), axis=0)
        i = 2
        while (i < len(contours)):
            pts = np.concatenate((pts, contours[i].vertices), axis=0)
            i = i + 1
        return pts

    def combineArea(contours):
        area = 0
        for contour in contours:
            area = area + contour.area
        return area

# Drawing functions

'''

    Default drawing colors (and there BGR values):
    Contours - pink (203, 192, 255)
    Contour Verticies - red (0, 0, 255)
    Contour Midpoint - (0, 100, 0)
    Reference Vector - green (0, 255, 0)
    Contour Label - blue (255, 0, 0)
    Rotated Bounding Box - cyan (255, 191, 0)
    Straight Bounding Box - yellow (0, 255, 255)

'''

# This is the default font, make sure to import this as well, or you can select your own
font = cv.FONT_HERSHEY_SIMPLEX

def drawContours(img, contours):
    contours_temp = []
    for contour in contours:
        if (isinstance(contour, ContourGroup)):
            if contour.useConvexHull:
                contours_temp.append(contour.convexHull)
            else:
                group_contours = contour.contours
                for contour in group_contours:
                    contours_temp.append(contour.points)
        else:
            if contour.useConvexHull:
                contours_temp.append(contour.convexHull)
            else:
                contours_temp.append(contour.points)

    img_temp = np.array(img, copy=True)
    dst = cv.drawContours(img_temp, contours_temp, -1, (203, 192, 255), 2)
    return dst

def drawBoundingBoxes(img, contours):
    for contour in contours:
        cv.drawContours(img, contour.rotatedRect, 0, (255, 191, 0), 2)
        cv.rectangle(img, contour.boundingBoxUpperLeftPoint, contour.boundingBoxLowerRightPoint, (0, 255, 255), 2)

def labelContours(img, contours):
    for i in range(len(contours)):
        anchor = contours[i].boundingBoxUpperLeftPoint
        anchor = (anchor[0] - 20, anchor[1] - 5)
        cv.putText(img, str(i + 1), anchor, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv.LINE_AA)

def drawReferenceVector(img, contours):
    for contour in contours:
        point = contour.referenceVector.getPoint(18, True)
        cv.circle(img, contour.midpoint, 3, (0, 100, 0), 2, cv.LINE_AA)
        cv.line(img, contour.midpoint, point, (0, 255, 0), 2, cv.LINE_AA)

def labelVerticies(img, contours):
    pts = None
    for contour in contours:
        if contour.useConvexHull:
            pts = contour.convexHull
        else:
            pts = contour.vertices
        for x in range(len(pts)):
            cv.putText(img, str(x + 1), (int(pts[x][0][0]), int(pts[x][0][1])), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)

def drawPnPAxes(img, imgpts):
    origin = tuple(imgpts[0].ravel())
    img = cv.line(img, origin, tuple(imgpts[1].ravel()), (0,0,255), 3)
    img = cv.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 3)
    img = cv.line(img, origin, tuple(imgpts[3].ravel()), (0,255,255), 3)
    return img

# Various methods for finding and processing contours
def findContours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours

def processContours(contours, frameCenter, useConvexHull, numOfCorners):
    processed_contours = []
    for contour in contours:
        if (cv.contourArea(contour) > 20):
            cnt = Contour(contour, frameCenter, useConvexHull, numOfCorners)
            processed_contours.append(cnt)
    idec = []
    for contour in processed_contours:
        if (contour.rotatedRectArea > 20):
            idec.append(contour)
    return idec

# This is used by filterContours
def withinRange(val, range):
    # TO-DO: Clean up the notation here
    if (val > range[0] and val < range[1]):
        return True
    else:
        return False

def filterContours(contours, targetAreaRange, targetFullnessRange, aspectRatioRange, frameSize):
    filteredContours = []
    for contour in contours:
        cntTargetArea = contour.area / frameSize
        cntTargetFullness = contour.area / contour.rotatedRectArea
        cntAspectRatio = contour.boundingBoxAspectRatio

        withinTargetAreaRange = withinRange(cntTargetArea, targetAreaRange)
        withinTargetFullnessRange = withinRange(cntTargetFullness, targetFullnessRange)
        withinTargetAspectRatioRange = withinRange(cntAspectRatio, aspectRatioRange)

        if (withinTargetAreaRange and withinTargetFullnessRange and withinTargetAspectRatioRange):
            filteredContours.append(contour)

    if (len(filteredContours) == 0):
        return None
    
    return filteredContours

def sortContours(filteredContours, sortingMode):
    # left to right
    if (sortingMode == 'left'):
        return sorted(filteredContours, key=lambda cnt: cnt.midpoint[0])
    # right to left
    elif (sortingMode == 'right'):
        return sorted(filteredContours, key=lambda cnt: cnt.midpoint[0], reverse=True)
    # top to bottom
    elif (sortingMode == 'top'):
        return sorted(filteredContours, key=lambda cnt: cnt.midpoint[1])
    # bottom to top
    elif (sortingMode == 'bottom'):
        return sorted(filteredContours, key=lambda cnt: cnt.midpoint[1], reverse=True)
    # center outwards
    elif (sortingMode == 'center'):
        return sorted(filteredContours, key=lambda cnt: cnt.distanceToCenter)

# TO-DO: Try to re-write this so it has a running time of O(N * log(N))
def pairContours(sortedContours, intersectionLocation, targetAreaRange, targetFullnessRange, aspectRatioRange, sortingMode, frameCenter, frameSize, useConvexHull, numOfPairCorners):
    pairs = []
    if (intersectionLocation == 'neither'):
        for i in range(len(sortedContours)):
            refContour = sortedContours[i]
            j = i + 1
            while (j < len(sortedContours)):
                contour = sortedContours[j]
                _contours = [refContour, contour]
                pair = ContourGroup(_contours, frameCenter, useConvexHull, numOfPairCorners)
                pairs.append(pair)
                j = j + 1
    else:
        for i in range(len(sortedContours)):
            refContour = sortedContours[i]
            j = i + 1
            while (j < len(sortedContours)):
                contour = sortedContours[j]
                refContourRefVector = refContour.contourLine
                contourRefVector = contour.contourLine
                intersectionPoint = refContourRefVector.intersects(contourRefVector)
                if (intersectionPoint is not None):
                    intersectionPoint[1] = frameCenter[1] * 2 - intersectionPoint[1]
                    if (intersectionLocation == 'above' and intersectionPoint[1] < refContour.midpoint[1] and intersectionPoint[1] < contour.midpoint[1]):
                        _contours = [refContour, contour]
                        pair = ContourGroup(_contours, frameCenter, useConvexHull, numOfPairCorners)
                        pairs.append(pair)
                    elif (intersectionLocation == 'below' and intersectionPoint[1] > refContour.midpoint[1] and intersectionPoint[1] > contour.midpoint[1]):
                        _contours = [refContour, contour]
                        pair = ContourGroup(_contours, frameCenter, useConvexHull, numOfPairCorners)
                        pairs.append(pair)
                    elif (intersectionLocation == 'right' and intersectionPoint[0] > refContour.midpoint[0] and intersectionPoint[0] > contour.midpoint[0]):
                        _contours = [refContour, contour]
                        pair = ContourGroup(_contours, frameCenter, useConvexHull, numOfPairCorners)
                        pairs.append(pair)
                    elif (intersectionLocation == 'left' and intersectionPoint[0] < refContour.midpoint[0] and intersectionPoint[0] < contour.midpoint[0]):
                        _contours = [refContour, contour]
                        pair = ContourGroup(_contours, frameCenter, useConvexHull, numOfPairCorners)
                        pairs.append(pair)
                j = j + 1
    # Now filter the pairs
    filteredPairs = filterContours(pairs, targetAreaRange, targetFullnessRange, aspectRatioRange, frameSize)

    if filteredPairs is None:
        return None

    # Now sort the pairs
    sortedPairs = sortContours(filteredPairs, sortingMode)

    # Return the first pair in the list, which theoretically is the closest thing to what we want
    return [sortedPairs[0]]