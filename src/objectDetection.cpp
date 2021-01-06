#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
//#include <opencv2/opencv.hpp>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>

#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"

/*
#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;
*/

typedef const boost::function< void(const sensor_msgs::ImageConstPtr&)> callback;

class objectDetector{
public:
    cv::Mat matImage01;
    cv::Mat matImage02;
    cv::Mat matImage03;
    sensor_msgs::ImagePtr messagePtr01;
    sensor_msgs::ImagePtr messagePtr02;
    sensor_msgs::ImagePtr messagePtr03;
    void imageCallback01(const sensor_msgs::ImageConstPtr& msg);
    void imageCallback02(const sensor_msgs::ImageConstPtr& msg);
    void imageCallback03(const sensor_msgs::ImageConstPtr& msg);
    objectDetector();
    ~objectDetector();
};

objectDetector::objectDetector() {
}

objectDetector::~objectDetector() {
}

static void wrapper_imageCallback01(void* pt2Object, const sensor_msgs::ImageConstPtr& msg);
static void wrapper_imageCallback02(void* pt2Object, const sensor_msgs::ImageConstPtr& msg);
static void wrapper_imageCallback03(void* pt2Object, const sensor_msgs::ImageConstPtr& msg);

void objectDetector::imageCallback01(const sensor_msgs::ImageConstPtr& msg) {
    try {
        this->matImage01 = cv_bridge::toCvShare(msg, "bgr8")->image;
        //---------------object_detect(cvimage)
        //cv::imshow("view", this->matImage);
        //cv::waitKey(30);

        this->messagePtr01 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->matImage01).toImageMsg();
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void objectDetector::imageCallback02(const sensor_msgs::ImageConstPtr& msg) {
    try {
        this->matImage02 = cv_bridge::toCvShare(msg, "bgr8")->image;
        //---------------object_detect(cvimage)
        //cv::imshow("view", this->matImage);
        //cv::waitKey(30);

        this->messagePtr02 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->matImage02).toImageMsg();
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void objectDetector::imageCallback03(const sensor_msgs::ImageConstPtr& msg) {
    try {
        this->matImage03 = cv_bridge::toCvShare(msg, "bgr8")->image;
        //---------------object_detect(cvimage)
        //cv::imshow("view", this->matImage);
        //cv::waitKey(30);

        this->messagePtr03 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->matImage03).toImageMsg();
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

static void wrapper_imageCallback01(void* pt2Object, const sensor_msgs::ImageConstPtr& msg) {
    objectDetector* det = (objectDetector*) pt2Object;
    det->imageCallback01(msg);
}

static void wrapper_imageCallback02(void* pt2Object, const sensor_msgs::ImageConstPtr& msg) {
    objectDetector* det = (objectDetector*) pt2Object;
    det->imageCallback02(msg);
}

static void wrapper_imageCallback03(void* pt2Object, const sensor_msgs::ImageConstPtr& msg) {
    objectDetector* det = (objectDetector*) pt2Object;
    det->imageCallback03(msg);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "object_detection");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    objectDetector detector;
    callback boundImageCallback01 = boost::bind(&objectDetector::imageCallback01, &detector, _1);
    callback boundImageCallback02 = boost::bind(&objectDetector::imageCallback02, &detector, _1);
    callback boundImageCallback03 = boost::bind(&objectDetector::imageCallback03, &detector, _1);

    image_transport::Subscriber sub01 = it.subscribe("/Cam/Image_raw01",1,boundImageCallback01);
    image_transport::Subscriber sub02 = it.subscribe("/Cam/Image_raw02",1,boundImageCallback02);
    image_transport::Subscriber sub03 = it.subscribe("/Cam/Image_raw03",1,boundImageCallback03);

    image_transport::Publisher pub01 = it.advertise("images01", 1);
    image_transport::Publisher pub02 = it.advertise("images02", 1);
    image_transport::Publisher pub03 = it.advertise("images03", 1);



    while (nh.ok()) {

        pub01.publish(detector.messagePtr01);
        pub02.publish(detector.messagePtr02);
        pub03.publish(detector.messagePtr03);
        ros::spinOnce();
    }

    return 0;
}
