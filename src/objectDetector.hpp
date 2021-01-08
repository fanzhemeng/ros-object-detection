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
#include "yolov5sEngine.hpp"


const char* subTopic01 = "/Cam/Image_raw01";
const char* subTopic02 = "/Cam/Image_raw02";
const char* subTopic03 = "/Cam/Image_raw03";
const char* pubTopic01 = "/detection01";
const char* pubTopic02 = "/detection02";
const char* pubTopic03 = "/detection03";
const char* pubTopic04 = "/detection_image01";
const char* pubTopic05 = "/detection_image02";
const char* pubTopic06 = "/detection_image03";


class objectDetector{
public:
    ros::Subscriber sub01;
    ros::Subscriber sub02;
    ros::Subscriber sub03;
    ros::Publisher pub01;
    ros::Publisher pub02;
    ros::Publisher pub03;
    ros::Publisher pub04;
    ros::Publisher pub05;
    ros::Publisher pub06;
    void imageCallback01(const sensor_msgs::ImageConstPtr& msg);
    void imageCallback02(const sensor_msgs::ImageConstPtr& msg);
    void imageCallback03(const sensor_msgs::ImageConstPtr& msg);
    objectDetector(ros::NodeHandle& nh);
private:
    static yolov5sengine yoloengine;
    sensor_msgs::ImageConstPtr msgIn01;
    sensor_msgs::ImageConstPtr msgIn02;
    sensor_msgs::ImageConstPtr msgIn03;
    cv::Mat cvImage01;
    cv::Mat cvImage02;
    cv::Mat cvImage03;
    bool check01;
    bool check02;
    bool check03;
    waytous_perception_msgs::ObjectArray msgOut01;
    waytous_perception_msgs::ObjectArray msgOut02;
    waytous_perception_msgs::ObjectArray msgOut03;
};

objectDetector::objectDetector(ros::NodeHandle& nh) {
    check01 = check02 = check03 = false;

    pub01 = nh.advertise<waytous_perception_msgs::ObjectArray>(pubTopic01, 1);
    pub02 = nh.advertise<waytous_perception_msgs::ObjectArray>(pubTopic02, 1);
    pub03 = nh.advertise<waytous_perception_msgs::ObjectArray>(pubTopic03, 1);
    pub04 = nh.advertise<sensor_msgs::Image>(pubTopic04, 1);
    pub05 = nh.advertise<sensor_msgs::Image>(pubTopic05, 1);
    pub06 = nh.advertise<sensor_msgs::Image>(pubTopic06, 1);

    sub01 = nh.subscribe(subTopic01, 1, &objectDetector::imageCallback01, this);
    sub02 = nh.subscribe(subTopic02, 1, &objectDetector::imageCallback02, this);
    sub03 = nh.subscribe(subTopic03, 1, &objectDetector::imageCallback03, this);

}

yolov5sengine objectDetector::yoloengine;

void objectDetector::imageCallback01(const sensor_msgs::ImageConstPtr& msg) {
    try {

    	this->msgIn01 = msg;
        this->cvImage01 = cv_bridge::toCvShare(msg, "bgr8")->image;
        this->check01 = true;

    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void objectDetector::imageCallback02(const sensor_msgs::ImageConstPtr& msg) {
    try {

    	this->msgIn02 = msg;
        this->cvImage02 = cv_bridge::toCvShare(msg, "bgr8")->image;
        this->check02 = true;

    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void objectDetector::imageCallback03(const sensor_msgs::ImageConstPtr& msg) {
    try {

    	this->msgIn03 = msg;
        this->cvImage03 = cv_bridge::toCvShare(msg, "bgr8")->image;
        this->check03 = true;

    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
    
    
    if (check01 && check02 && check03) {
        // --------detect using yoloengine, with batch_size=3
        std::vector<cv::Mat> cvimgs;
        cvimgs.push_back(this->cvImage01);
        cvimgs.push_back(this->cvImage02);
        cvimgs.push_back(this->cvImage03);

        std::vector<std::vector<Yolo::Detection>> batch_res = yoloengine.detect(cvimgs);

        cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->cvImage01).toImageMsg((this->msgOut01).raw_image);
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->cvImage02).toImageMsg((this->msgOut02).raw_image);
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->cvImage03).toImageMsg((this->msgOut03).raw_image);

        this->msgOut01.header = (this->msgIn01)->header;
        this->msgOut02.header = (this->msgIn02)->header;
        this->msgOut03.header = (this->msgIn03)->header;

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
                	this->msgOut01.foreground_objects.push_back(obj);
                }
                else if (b==1) {
                	this->msgOut02.foreground_objects.push_back(obj);
                }
                else if (b==2) {
                	this->msgOut03.foreground_objects.push_back(obj);
                }
            }
        }
        this->pub01.publish(this->msgOut01);
        this->pub02.publish(this->msgOut02);
        this->pub03.publish(this->msgOut03);
        this->pub04.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvimgs[0]).toImageMsg());
        this->pub05.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvimgs[1]).toImageMsg());
        this->pub06.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvimgs[2]).toImageMsg());
        
        check01 = check02 = check03 = false;
    }
}
#endif

