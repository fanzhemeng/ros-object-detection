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
#include <waytous_perception_msgs/Object.h>
#include <waytous_perception_msgs/ObjectArray.h>
#include <waytous_perception_msgs/Rect.h>


#include <iostream>
#include <fstream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"


#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 3

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;


typedef const boost::function< void(const sensor_msgs::ImageConstPtr&)> callback;
const char* topic01 = "/Cam/Image_raw01";
const char* topic02 = "/Cam/Image_raw02";
const char* topic03 = "/Cam/Image_raw03";

class yolov5sengine{
public:
    std::string engine_name = "yolov5s.engine";
    std::vector<std::string> classes;

    float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    void* buffers[2];
    int inputIndex;
    int outputIndex;
    cudaStream_t stream;
    yolov5sengine();
    ~yolov5sengine();
    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize);
    std::vector<std::vector<Yolo::Detection>> detect(std::vector<cv::Mat>& imgs);
};

yolov5sengine::yolov5sengine() {

    // get classes names from classes.txt
    std::ifstream infile("classes.txt");
    std::string line;
    while (std::getline(infile, line)) { classes.push_back(line); }

    cudaSetDevice(DEVICE);
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // prepare input data
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    CHECK(cudaStreamCreate(&stream));
}


yolov5sengine::~yolov5sengine() {
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

void yolov5sengine::doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

std::vector<std::vector<Yolo::Detection>> yolov5sengine::detect(std::vector<cv::Mat>& imgs) {
    int fcount = BATCH_SIZE;
    for (int b=0; b<fcount; b++) {
        cv::Mat pr_img = preprocess_img(imgs[b]);
        int i=0;
        for (int row = 0; row < INPUT_H; ++row) {
            uchar* uc_pixel = pr_img.data + row * pr_img.step;
            for (int col = 0; col < INPUT_W; ++col) {
                data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }
    }

    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
    for (int b = 0; b < fcount; b++) {
        auto& res = batch_res[b];
        nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
    }

    for (int b = 0; b < fcount; b++) {
        auto& res = batch_res[b];
        std::cout << std::to_string(b) << " found: " << res.size() << std::endl;
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(imgs[b], res[j].bbox);
            //cv::rectangle(imgs[b], r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            std::string classname = classes[(int)res[j].class_id];
            //cv::putText(imgs[b], classname, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0x27, 0xC1, 0x36), 2);
        }
    }
    return batch_res;
}


class objectDetector{
public:
    static yolov5sengine yoloengine;
    //cv::Mat matImage01;
    //cv::Mat matImage02;
    //cv::Mat matImage03;
    cv::Mat matImage[3];
    //bool check01 = false;
    //bool check02 = false;
    //bool check03 = false;
    bool check[3];
    //waytous_perception_msgs::ObjectArray messagePtr01;
    //waytous_perception_msgs::ObjectArray messagePtr02;
    //waytous_perception_msgs::ObjectArray messagePtr03;
    waytous_perception_msgs::ObjectArray messagePtr[3];
    void imageCallback(const sensor_msgs::ImageConstPtr& msg, const std::string &topic);
    objectDetector();
};

objectDetector::objectDetector() {
    check[0] = check[1] = check[2] = false;
}

yolov5sengine objectDetector::yoloengine;

static void wrapper_imageCallback(void* pt2Object, const sensor_msgs::ImageConstPtr& msg, const std::string &topic);


void objectDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg, const std::string &topic) {
    try {
        if (topic == "/Cam/Image_raw01") {
            this->matImage[0] = cv_bridge::toCvShare(msg, "bgr8")->image;
            this->check[0] = true;
        } else if (topic == "/Cam/Image_raw02") {
            this->matImage[1] = cv_bridge::toCvShare(msg, "bgr8")->image;
            this->check[1] = true;
        } else {
            this->matImage[2] = cv_bridge::toCvShare(msg, "bgr8")->image;
            this->check[2] = true;
        }
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
    
    
    if (check[0] && check[1] && check[2]) {
        // --------detect using yoloengine, with batch_size=3
        std::vector<cv::Mat> cvimgs;
        cvimgs.push_back(this->matImage[0]);
        cvimgs.push_back(this->matImage[1]);
        cvimgs.push_back(this->matImage[2]);

        std::vector<std::vector<Yolo::Detection>> batch_res = yoloengine.detect(cvimgs);
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
                this->messagePtr[b].foreground_objects.push_back(obj);
            }
            cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvimgs[b]).toImageMsg((this->messagePtr[b]).roi_image);
        }
        //cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvimgs[0]).toImageMsg((this->messagePtr01).roi_image);
        //cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvimgs[1]).toImageMsg((this->messagePtr02).roi_image);
        //cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvimgs[2]).toImageMsg((this->messagePtr03).roi_image);

        check[0] = check[1] = check[2] = false;
    }
}


static void wrapper_imageCallback(void* pt2Object, const sensor_msgs::ImageConstPtr& msg, const std::string &topic) {
    objectDetector* det = (objectDetector*) pt2Object;
    det->imageCallback(msg, topic);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "object_detection");
    ros::NodeHandle nh;
    //image_transport::ImageTransport it(nh);

    objectDetector detector;
    callback boundImageCallback01 = boost::bind(&objectDetector::imageCallback, &detector, _1, topic01);
    callback boundImageCallback02 = boost::bind(&objectDetector::imageCallback, &detector, _1, topic02);
    callback boundImageCallback03 = boost::bind(&objectDetector::imageCallback, &detector, _1, topic03);

    ros::Subscriber sub01 = nh.subscribe(topic01, 1,boundImageCallback01);
    ros::Subscriber sub02 = nh.subscribe(topic02, 1,boundImageCallback02);
    ros::Subscriber sub03 = nh.subscribe(topic03, 1,boundImageCallback03);

    ros::Publisher pub01 = nh.advertise<waytous_perception_msgs::ObjectArray>("detection01", 1);
    ros::Publisher pub02 = nh.advertise<waytous_perception_msgs::ObjectArray>("detection02", 1);
    ros::Publisher pub03 = nh.advertise<waytous_perception_msgs::ObjectArray>("detection03", 1);



    while (nh.ok()) {

        pub01.publish(detector.messagePtr[0]);
        pub02.publish(detector.messagePtr[1]);
        pub03.publish(detector.messagePtr[2]);
        ros::spinOnce();
    }

    return 0;
}
