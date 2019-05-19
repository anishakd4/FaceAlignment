#include<dlib/image_processing.h>
#include<dlib/image_processing/frontal_face_detector.h>
#include<dlib/opencv.h>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include<iostream>

using namespace dlib;
using namespace cv;
using namespace std;

//similarity transform given two pairs of corresponding points. OpenCV requires 3 points for calculating similarity matrix.
// We are assuming the third point as the third point of the eqillateral triangle with these two given points.
void similarityTransformMat(std::vector<Point2f> initialPoints, std::vector<Point2f> destinationPoints, Mat &similarityMat){
    double sin60 = sin(60* M_PI/180.0);
    double cos60 = cos(60* M_PI/180.0);

    //third point is caluculated for initial points
    double initialPointsX =  cos60 * (initialPoints[0].x - initialPoints[1].x) - sin60 * (initialPoints[0].y - initialPoints[1].y) + initialPoints[1].x;
    double initialPointsY =  sin60 * (initialPoints[0].x - initialPoints[1].x) + cos60 * (initialPoints[0].y - initialPoints[1].y) + initialPoints[1].y;

    //third point is caluculated for destination points
    double destinationPointsX =  cos60 * (destinationPoints[0].x - destinationPoints[1].x) - sin60 * (destinationPoints[0].y - destinationPoints[1].y) 
        + destinationPoints[1].x;
    double destinationPointsY =  sin60 * (destinationPoints[0].x - destinationPoints[1].x) + cos60 * (destinationPoints[0].y - destinationPoints[1].y) 
        + destinationPoints[1].y;

    initialPoints.push_back(Point2f(initialPointsX, initialPointsY));
    destinationPoints.push_back(Point2f(destinationPointsX, destinationPointsY));
    
    //calculate similarity transform
    similarityMat = estimateAffinePartial2D(initialPoints, destinationPoints);
}

//face Alignes a facial image to a standard size. The normalization is done based on Dlib's landmark points.
//After the normalization the left corner of the left eye is at (0.3*w, h/3) and the right corner of the right eye 
//is at (0.7*w, h/3) where w and h are the width and height of standard size.
void faceAlign(Mat &faceAligned, cv::Size size, Mat image, std::vector<Point2f> facelandmarks){

    std::vector<Point2f> initialPoints;
    std::vector<Point2f> destinationPoints;

    if(facelandmarks.size() == 68){
        //location of left eye left corner in input image
        initialPoints.push_back(facelandmarks[36]);

        //location of right eye right corner in input image
        initialPoints.push_back(facelandmarks[45]);
    }

    //location of left eye left corner in faceAligned image
    Point2f point1 = Point2f(0.3*size.width, size.height/3);

    //location of right eye right corner in faceAligned image
    Point2f point2 = Point2f(0.7*size.width, size.height/3);

    destinationPoints.push_back(point1);
    destinationPoints.push_back(point2);

    //calculate similarity transform
    Mat similarityMat;
    similarityTransformMat(initialPoints, destinationPoints, similarityMat);

    //define faceAligned image
    faceAligned = Mat::zeros(600, 600, CV_32FC3);

    //apply similarity transform to input image
    warpAffine(image, faceAligned, similarityMat, faceAligned.size());

    //apply similarity transform to landmark points if needed
    transform(facelandmarks, facelandmarks, similarityMat);
}

//finds face landmark points
std::vector<Point2f> getFaceLandmarks(Mat image, frontal_face_detector faceDetector, 
    shape_predictor landmarkDetector){

        //vector to store face landmark points
        std::vector<Point2f> points;

        //convert image to dlib image format
        cv_image<bgr_pixel> dlibImage(image);

        //detect faces in the image
        std::vector<dlib::rectangle> faces = faceDetector(dlibImage);

        //go through first face in the image
        if(faces.size() > 0){

            //get the first face rectangle
            dlib::rectangle firstFace = faces[0];

            //get face landmarks
            dlib::full_object_detection faceLandmarks = landmarkDetector(dlibImage, firstFace);

            //convert full_object_detection to vector form
            for(int i=0; i< faceLandmarks.num_parts(); i++){
                Point2f point(faceLandmarks.part(i).x(), faceLandmarks.part(i).y());
                points.push_back(point);
            }
        }

        return points;
}

int main(){

    //Read input image
    Mat image = cv::imread("../assets/anish.jpg");

    //check if image exists
    if(image.empty()){
        cout<<"can not find image"<<endl;
        return 0;
    }

    //Define face detector
    frontal_face_detector faceDetector = get_frontal_face_detector();

    //define landmark detector
    shape_predictor landmarkDetector;

    //load face landmark model
    deserialize("../dlibAndModel/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

    //get face landmarks
    std::vector<Point2f> faceLandmarks = getFaceLandmarks(image, faceDetector, landmarkDetector);

    //convert image to float point and in the range 0 and 1
    image.convertTo(image, CV_32FC3, 1/255.0);

    //define faceAligned image
    Mat faceAligned;

    //size of faceAligned image. Specify the size of the aligned face image
    Size size(600, 600);

    //align face image
    faceAlign(faceAligned, size, image, faceLandmarks);

    //convert the images back to CV_8UC3
    image.convertTo(image, CV_8UC3, 255);
    faceAligned.convertTo(faceAligned, CV_8UC3, 255);

    //Create windows to display images
    namedWindow("image", WINDOW_NORMAL);
    namedWindow("face Aligned", WINDOW_NORMAL);

    //display images
    imshow("image", image);
    imshow("face Aligned", faceAligned);

    //press esc to exit the program
    waitKey(0);

    //close all the opened windows
    destroyAllWindows();

    return 0;
}