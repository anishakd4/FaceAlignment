import cv2, sys, dlib, math
import numpy as np

#similarity transform given two pairs of corresponding points. OpenCV requires 3 points for calculating similarity matrix.
#We are assuming the third point as the third point of the eqillateral triangle with these two given points.
def similarityTransformMat(initialPoints, destinationPoints):
        sin60 = math.sin(60*math.pi / 180)
        cos60 = math.cos(60*math.pi / 180)

        #third point is caluculated for initial points
        xin = cos60*(initialPoints[0][0] - initialPoints[1][0]) - sin60*(initialPoints[0][1] - initialPoints[1][1]) + initialPoints[1][0]
        yin = sin60*(initialPoints[0][0] - initialPoints[1][0]) + cos60*(initialPoints[0][1] - initialPoints[1][1]) + initialPoints[1][1]

        initialPoints.append([np.int(xin), np.int(yin)])

        #third point is caluculated for destination points
        xout = cos60*(destinationPoints[0][0] - destinationPoints[1][0]) - sin60*(destinationPoints[0][1] - destinationPoints[1][1]) + destinationPoints[1][0]
        yout = sin60*(destinationPoints[0][0] - destinationPoints[1][0]) + cos60*(destinationPoints[0][1] - destinationPoints[1][1]) + destinationPoints[1][1]

        destinationPoints.append([np.int(xout), np.int(yout)])

        # calculate similarity transform.
        tform = cv2.estimateAffinePartial2D(np.array([initialPoints]), np.array([destinationPoints]))
        return tform[0]

#face Alignes a facial image to a standard size. The normalization is done based on Dlib's landmark points.
#After the normalization the left corner of the left eye is at (0.3*w, h/3) and the right corner of the right eye 
#is at (0.7*w, h/3) where w and h are the width and height of standard size.
def faceAlign(image, size, faceLandmarks):
        (h, w) = size
        initialPoints = []
        destinationPoints = []

        #location of left eye left corner and right eye right corner in input image
        initialPoints = [faceLandmarks[36], faceLandmarks[45]]

        #location of left eye left corner and right eye right corner in face aligned image
        destinationPoints = [(np.int(0.3*w), np.int(h/3)), (np.int(0.7*w), np.int(h/3))]

        #calculate similarity transform
        similarityTransform = similarityTransformMat(initialPoints, destinationPoints)

        #define faceAligned image
        faceAligned = np.zeros((image.shape), dtype=image.dtype)

        #apply similarity transform
        faceAligned = cv2.warpAffine(image, similarityTransform, (w, h))

        return faceAligned


#find face landmarks
def getFaceLandmarks(image, faceDetector, landmarkDetector):

        #define to store face landmarks
        points = []

        #convert image to dlib image format
        dlibImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #detect faces
        faces = faceDetector(dlibImage, 0)

        #go through first face in the image
        if(len(faces) > 0):
                faceRectangle = faces[0]
                dlibRectangle = dlib.rectangle(int(faceRectangle.left()), int(faceRectangle.top()), 
                        int(faceRectangle.right()), int(faceRectangle.bottom()))

                faceLandmarks = landmarkDetector(image, dlibRectangle)

                for part in faceLandmarks.parts():
                        points.append((part.x, part.y))

        return points


#Read image
image = cv2.imread("../assets/anish.jpg")

#check if image exists
if image is None:
        print("can not find image")
        sys.exit()

#define face detector
faceDetector = dlib.get_frontal_face_detector()

#define landmark detector and load face landmark model
landmarkDetector = dlib.shape_predictor("../dlibAndModel/shape_predictor_68_face_landmarks.dat")

#get face landmarks
faceLandmarks = getFaceLandmarks(image, faceDetector, landmarkDetector)

#convert to numpy array
faceLandmarks = np.array(faceLandmarks)

#convert image to floating point and in the range 0 to 1
image = np.float32(image)/255.0

#size of the faceAligned image
size = (600, 600)

#align face image
faceAligned = faceAlign(image, size, faceLandmarks)

#create windows to display images
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.namedWindow("face aligned", cv2.WINDOW_NORMAL)

#display images
cv2.imshow("image", image)
cv2.imshow("face aligned", faceAligned)

#press esc to exit the program
cv2.waitKey(0)

#close all the opened windows
cv2.destroyAllWindows()