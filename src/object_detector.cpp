// #include <fstream>
// #include <vector>
// #include <opencv2/opencv.hpp>
// #include <ros/ros.h>
// #include <image_transport/image_transport.h>
// #include <cv_bridge/cv_bridge.h>
// #include <sensor_msgs/Image.h>
// #include <waytous_perception_msgs/Object.h>
// #include <waytous_perception_msgs/ObjectArray.h>
// #include <waytous_perception_msgs/Rect.h>
// #include "yolov5s_engine.hpp"
#include "object_detector.h"


Yolov5sEngine ObjectDetector::mYoloEngine;

ObjectDetector::ObjectDetector(ros::NodeHandle& nh) {
    mCheck01 = mCheck02 = mCheck03 = mCheck04 = false;

    mPub01 = nh.advertise<waytous_perception_msgs::ObjectArray>(pubTopic01, 1);
    mPub02 = nh.advertise<waytous_perception_msgs::ObjectArray>(pubTopic02, 1);
    mPub03 = nh.advertise<waytous_perception_msgs::ObjectArray>(pubTopic03, 1);
    mPub04 = nh.advertise<waytous_perception_msgs::ObjectArray>(pubTopic04, 1);
    mPub05 = nh.advertise<sensor_msgs::Image>(pubTopic05, 1);
    mPub06 = nh.advertise<sensor_msgs::Image>(pubTopic06, 1);
    mPub07 = nh.advertise<sensor_msgs::Image>(pubTopic07, 1);
    mPub08 = nh.advertise<sensor_msgs::Image>(pubTopic08, 1);

    mSub01 = nh.subscribe(subTopic01, 1, &ObjectDetector::imageCallback01, this);
    mSub02 = nh.subscribe(subTopic02, 1, &ObjectDetector::imageCallback02, this);
    mSub03 = nh.subscribe(subTopic03, 1, &ObjectDetector::imageCallback03, this);
    mSub04 = nh.subscribe(subTopic04, 1, &ObjectDetector::imageCallback04, this);

}

void ObjectDetector::imageCallback01(const sensor_msgs::ImageConstPtr& msg) {
    try {

    	this->mMsgIn01 = msg;
        this->mCvImage01 = cv_bridge::toCvShare(msg, "bgr8")->image;
        this->mCheck01 = true;

    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void ObjectDetector::imageCallback02(const sensor_msgs::ImageConstPtr& msg) {
    try {

    	this->mMsgIn02 = msg;
        this->mCvImage02 = cv_bridge::toCvShare(msg, "bgr8")->image;
        this->mCheck02 = true;

    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void ObjectDetector::imageCallback03(const sensor_msgs::ImageConstPtr& msg) {
    try {

    	this->mMsgIn03 = msg;
        this->mCvImage03 = cv_bridge::toCvShare(msg, "bgr8")->image;
        this->mCheck03 = true;

    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void ObjectDetector::imageCallback04(const sensor_msgs::ImageConstPtr& msg) {
    try {

    	this->mMsgIn04 = msg;
        this->mCvImage04 = cv_bridge::toCvShare(msg, "bgr8")->image;
        this->mCheck04 = true;

    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
    
    
    if (mCheck01 && mCheck02 && mCheck03 && mCheck04) {
        // --------detect using mYoloEngine, with batch_size=3
        std::vector<cv::Mat> cvimgs;
        cvimgs.push_back(this->mCvImage01);
        cvimgs.push_back(this->mCvImage02);
        cvimgs.push_back(this->mCvImage03);
        cvimgs.push_back(this->mCvImage04);

        std::vector<std::vector<Yolo::Detection>> batch_res = this->mYoloEngine.detect(cvimgs);

        cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->mCvImage01).toImageMsg((this->mMsgOut01).raw_image);
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->mCvImage02).toImageMsg((this->mMsgOut02).raw_image);
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->mCvImage03).toImageMsg((this->mMsgOut03).raw_image);
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->mCvImage04).toImageMsg((this->mMsgOut04).raw_image);

        this->mMsgOut01.header = (this->mMsgIn01)->header;
        this->mMsgOut02.header = (this->mMsgIn02)->header;
        this->mMsgOut03.header = (this->mMsgIn03)->header;
        this->mMsgOut04.header = (this->mMsgIn04)->header;

        for (int b=0; b<batch_res.size(); b++) {
            auto& res = batch_res[b];
            for (size_t j = 0; j < res.size(); j++) {
                waytous_perception_msgs::Object obj;
                waytous_perception_msgs::Rect r;
                cv::Rect cv_rect = get_rect(cvimgs[b], res[j].bbox);
                r.x = cv_rect.x;
                r.y = cv_rect.y;
                r.w = cv_rect.width;
                r.h = cv_rect.height;
                obj.rect = r;
                obj.label_type = (int)res[j].class_id;
                obj.score = res[j].conf;

                if (b==0) {
                	this->mMsgOut01.foreground_objects.push_back(obj);
                }
                else if (b==1) {
                	this->mMsgOut02.foreground_objects.push_back(obj);
                }
                else if (b==2) {
                	this->mMsgOut03.foreground_objects.push_back(obj);
                }
                else if (b==3) {
                	this->mMsgOut04.foreground_objects.push_back(obj);
                }
            }
        }
        this->mPub01.publish(this->mMsgOut01);
        this->mPub02.publish(this->mMsgOut02);
        this->mPub03.publish(this->mMsgOut03);
        this->mPub04.publish(this->mMsgOut04);
        this->mPub05.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvimgs[0]).toImageMsg());
        this->mPub06.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvimgs[1]).toImageMsg());
        this->mPub07.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvimgs[2]).toImageMsg());
        this->mPub08.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvimgs[3]).toImageMsg());
        
        mCheck01 = mCheck02 = mCheck03 = mCheck04 = false;
    }
}
