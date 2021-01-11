#ifndef OBJECT_DETECTOR_H_
#define OBJECT_DETECTOR_H_

#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <waytous_perception_msgs/Object.h>
#include <waytous_perception_msgs/ObjectArray.h>
#include <waytous_perception_msgs/Rect.h>
#include "yolov5s_engine.hpp"


class ObjectDetector{
public:
    void imageCallback01(const sensor_msgs::ImageConstPtr& msg);
    void imageCallback02(const sensor_msgs::ImageConstPtr& msg);
    void imageCallback03(const sensor_msgs::ImageConstPtr& msg);
    void imageCallback04(const sensor_msgs::ImageConstPtr& msg);
    ObjectDetector(ros::NodeHandle& nh);
private:
    static Yolov5sEngine mYoloEngine;
    const char* subTopic01 = "/Cam/Image_raw01";
    const char* subTopic02 = "/Cam/Image_raw02";
    const char* subTopic03 = "/Cam/Image_raw03";
    const char* subTopic04 = "/Cam/Image_raw03";
    const char* pubTopic01 = "/detection01";
    const char* pubTopic02 = "/detection02";
    const char* pubTopic03 = "/detection03";
    const char* pubTopic04 = "/detection04";
    const char* pubTopic05 = "/detection_image01";
    const char* pubTopic06 = "/detection_image02";
    const char* pubTopic07 = "/detection_image03";
    const char* pubTopic08 = "/detection_image04";
    ros::Subscriber mSub01;
    ros::Subscriber mSub02;
    ros::Subscriber mSub03;
    ros::Subscriber mSub04;
    ros::Publisher mPub01;
    ros::Publisher mPub02;
    ros::Publisher mPub03;
    ros::Publisher mPub04;
    ros::Publisher mPub05;
    ros::Publisher mPub06;
    ros::Publisher mPub07;
    ros::Publisher mPub08;
    sensor_msgs::ImageConstPtr mMsgIn01;
    sensor_msgs::ImageConstPtr mMsgIn02;
    sensor_msgs::ImageConstPtr mMsgIn03;
    sensor_msgs::ImageConstPtr mMsgIn04;
    cv::Mat mCvImage01;
    cv::Mat mCvImage02;
    cv::Mat mCvImage03;
    cv::Mat mCvImage04;
    bool mCheck01;
    bool mCheck02;
    bool mCheck03;
    bool mCheck04;
    waytous_perception_msgs::ObjectArray mMsgOut01;
    waytous_perception_msgs::ObjectArray mMsgOut02;
    waytous_perception_msgs::ObjectArray mMsgOut03;
    waytous_perception_msgs::ObjectArray mMsgOut04;
};
#endif
