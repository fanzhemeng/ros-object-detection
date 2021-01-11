#ifndef OBJECT_DETECTOR_H_
#define OBJECT_DETECTOR_H_

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
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
    ~ObjectDetector();
private:
    static Yolov5sEngine mYoloEngine;
    const char* mSubTopic01 = "/cam_01/data";
    const char* mSubTopic02 = "/cam_02/data";
    const char* mSubTopic03 = "/cam_03/data";
    const char* mSubTopic04 = "/cam_04/data";
    const char* mPubTopic01 = "/detection01";
    const char* mPubTopic02 = "/detection02";
    const char* mPubTopic03 = "/detection03";
    const char* mPubTopic04 = "/detection04";
    const char* mPubTopic05 = "/detection_image01";
    const char* mPubTopic06 = "/detection_image02";
    const char* mPubTopic07 = "/detection_image03";
    const char* mPubTopic08 = "/detection_image04";
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
    std::queue<sensor_msgs::ImageConstPtr> mMsgsIn01;
    std::queue<sensor_msgs::ImageConstPtr> mMsgsIn02;
    std::queue<sensor_msgs::ImageConstPtr> mMsgsIn03;
    std::queue<sensor_msgs::ImageConstPtr> mMsgsIn04;
    std::mutex mMtx01;
    std::mutex mMtx02;
    std::mutex mMtx03;
    std::mutex mMtx04;
    std::vector<cv::Mat> mCvImagesIn;
    std::vector<cv::Mat> mCvImagesOut;
    waytous_perception_msgs::ObjectArray mMsgOut01;
    waytous_perception_msgs::ObjectArray mMsgOut02;
    waytous_perception_msgs::ObjectArray mMsgOut03;
    waytous_perception_msgs::ObjectArray mMsgOut04;
    sensor_msgs::ImageConstPtr mMsgOut05;
    sensor_msgs::ImageConstPtr mMsgOut06;
    sensor_msgs::ImageConstPtr mMsgOut07;
    sensor_msgs::ImageConstPtr mMsgOut08;
    std::thread* mWorkThread = nullptr;
    bool mIsWorking = true;
};
#endif

