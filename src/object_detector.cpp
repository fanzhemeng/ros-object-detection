//#include <vector>
//#include <opencv2/opencv.hpp>
//#include <ros/ros.h>
//#include <cv_bridge/cv_bridge.h>
//#include <sensor_msgs/Image.h>
//#include <waytous_perception_msgs/Object.h>
//#include <waytous_perception_msgs/ObjectArray.h>
//#include <waytous_perception_msgs/Rect.h>
//#include "yolov5s_engine.hpp"
#include "object_detector.hpp"


ObjectDetector::ObjectDetector(ros::NodeHandle& nh) {

    mPub01 = nh.advertise<waytous_perception_msgs::ObjectArray>(mPubTopic01, 1);
    mPub02 = nh.advertise<waytous_perception_msgs::ObjectArray>(mPubTopic02, 1);
    mPub03 = nh.advertise<waytous_perception_msgs::ObjectArray>(mPubTopic03, 1);
    mPub04 = nh.advertise<waytous_perception_msgs::ObjectArray>(mPubTopic04, 1);
    mPub05 = nh.advertise<sensor_msgs::Image>(mPubTopic05, 1);
    mPub06 = nh.advertise<sensor_msgs::Image>(mPubTopic06, 1);
    mPub07 = nh.advertise<sensor_msgs::Image>(mPubTopic07, 1);
    mPub08 = nh.advertise<sensor_msgs::Image>(mPubTopic08, 1);

    mSub01 = nh.subscribe(this->mSubTopic01, 1, &ObjectDetector::imageCallback01, this);
    mSub02 = nh.subscribe(this->mSubTopic02, 1, &ObjectDetector::imageCallback02, this);
    mSub03 = nh.subscribe(this->mSubTopic03, 1, &ObjectDetector::imageCallback03, this);
    mSub04 = nh.subscribe(this->mSubTopic04, 1, &ObjectDetector::imageCallback04, this);

    mYoloEngine.setup();
    
    mIsWorking = true;
    mWorkThread = new std::thread([this]{

// proc data
while(mIsWorking)
{
    if ((!this->mMsgsIn01.empty()) && (!this->mMsgsIn02.empty()) && (!this->mMsgsIn03.empty()) && (!this->mMsgsIn04.empty())) {
        // --------detect using mYoloEngine, with batch_size=3
        auto start_process = std::chrono::system_clock::now();

        sensor_msgs::ImageConstPtr msgIn01, msgIn02, msgIn03, msgIn04;
        {
            std::unique_lock<std::mutex> lck(this->mMtx01);
            msgIn01 = this->mMsgsIn01.front();
            this->mMsgsIn01.pop();
        }
        {
            std::unique_lock<std::mutex> lck(this->mMtx02);
            msgIn02 = this->mMsgsIn02.front();
            this->mMsgsIn02.pop();
        }
        {
            std::unique_lock<std::mutex> lck(this->mMtx03);
            msgIn03 = this->mMsgsIn03.front();
            this->mMsgsIn03.pop();
        }
        {
            std::unique_lock<std::mutex> lck(this->mMtx04);
            msgIn04 = this->mMsgsIn04.front();
            this->mMsgsIn04.pop();
        }

        try {

            this->mCvImagesIn.push_back(cv_bridge::toCvShare(msgIn01, "bgr8")->image);
            this->mCvImagesIn.push_back(cv_bridge::toCvShare(msgIn02, "bgr8")->image);
            this->mCvImagesIn.push_back(cv_bridge::toCvShare(msgIn03, "bgr8")->image);
            this->mCvImagesIn.push_back(cv_bridge::toCvShare(msgIn04, "bgr8")->image);

        }
        catch (cv_bridge::Exception& e) {
            //ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
            ROS_ERROR("Could not convert from 'sensor_msgs::Image' to 'bgr8'.");
            mIsWorking = false;
            return;
        }

        for (int i=0; i<this->mCvImagesIn.size(); i++) {
            this->mCvImagesOut.push_back(this->mCvImagesIn[i]);
        }

        auto start_inf = std::chrono::system_clock::now();
        std::vector<std::vector<Yolo::Detection>> batch_res = this->mYoloEngine.detect(this->mCvImagesOut);
        auto end_inf = std::chrono::system_clock::now();
        std::cout << "engine detect time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_inf - start_inf).count() << "ms" << std::endl;

        this->mMsgOut01.header = msgIn01->header;
        this->mMsgOut02.header = msgIn02->header;
        this->mMsgOut03.header = msgIn03->header;
        this->mMsgOut04.header = msgIn04->header;

        cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->mCvImagesIn[0]).toImageMsg((this->mMsgOut01).raw_image);
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->mCvImagesIn[1]).toImageMsg((this->mMsgOut02).raw_image);
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->mCvImagesIn[2]).toImageMsg((this->mMsgOut03).raw_image);
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->mCvImagesIn[3]).toImageMsg((this->mMsgOut04).raw_image);

        for (int b=0; b<batch_res.size(); b++) {
            auto& res = batch_res[b];
            for (size_t j = 0; j < res.size(); j++) {
                waytous_perception_msgs::Object obj;
                waytous_perception_msgs::Rect r;
                cv::Rect cv_rect = get_rect(this->mCvImagesOut[b], res[j].bbox);
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

        this->mMsgOut05 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->mCvImagesOut[0]).toImageMsg();
        this->mMsgOut06 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->mCvImagesOut[1]).toImageMsg();
        this->mMsgOut07 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->mCvImagesOut[2]).toImageMsg();
        this->mMsgOut08 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->mCvImagesOut[3]).toImageMsg();

        this->mPub01.publish(this->mMsgOut01);
        this->mPub02.publish(this->mMsgOut02);
        this->mPub03.publish(this->mMsgOut03);
        this->mPub04.publish(this->mMsgOut04);
        this->mPub05.publish(this->mMsgOut05);
        this->mPub06.publish(this->mMsgOut06);
        this->mPub07.publish(this->mMsgOut07);
        this->mPub08.publish(this->mMsgOut08);

        auto end_process = std::chrono::system_clock::now();
        std::cout << "process total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_process - start_process).count() << "ms" << std::endl << std::endl;

    }
}
    });
}

ObjectDetector::~ObjectDetector()
{
    mIsWorking = false;
    if(mWorkThread)
    {
        mWorkThread->join();
        delete mWorkThread;
        mWorkThread = nullptr;
    }
}

Yolov5sEngine ObjectDetector::mYoloEngine;

void ObjectDetector::imageCallback01(const sensor_msgs::ImageConstPtr& msg) {

    std::unique_lock<std::mutex> lck(this->mMtx01);
    this->mMsgsIn01.push(msg);
}

void ObjectDetector::imageCallback02(const sensor_msgs::ImageConstPtr& msg) {

    std::unique_lock<std::mutex> lck(this->mMtx02);
    this->mMsgsIn02.push(msg);
}

void ObjectDetector::imageCallback03(const sensor_msgs::ImageConstPtr& msg) {

    std::unique_lock<std::mutex> lck(this->mMtx03);
    this->mMsgsIn03.push(msg);
}

void ObjectDetector::imageCallback04(const sensor_msgs::ImageConstPtr& msg) {

    std::unique_lock<std::mutex> lck(this->mMtx04);
    this->mMsgsIn04.push(msg);
}
