#include <ros/ros.h>
#include "objectDetector.hpp"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "object_detection_node");
    ros::NodeHandle nh;
    objectDetector detector(nh);

    ros::spin();

    return 0;
}
