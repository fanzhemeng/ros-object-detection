#include <ros/ros.h>
#include "object_detector.hpp"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "object_detection_node");
    ros::NodeHandle nh;
    ObjectDetector detector(nh);

    ros::spin();

    return 0;
}
