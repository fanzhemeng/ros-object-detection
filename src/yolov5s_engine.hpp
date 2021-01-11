#ifndef YOLOV5S_ENGINE_H_
#define YOLOV5S_ENGINE_H_

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
#define BATCH_SIZE 4

#define NET s  // s m l x
#define NETSTRUCT(str) createEngine_##str
#define CREATENET(net) NETSTRUCT(net)
#define STR1(x) #x
#define STR2(x) STR1(x)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
static Logger gLogger;


class Yolov5sEngine{
public:
    std::string filepath = "/home/huituo/catkin_ws/src/object_detection_with_msgs/";
    std::string engine_name = filepath+"yolov5s.engine";
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

    Yolov5sEngine() {};
    ~Yolov5sEngine();
    void setup();
    std::vector<std::vector<Yolo::Detection>> detect(std::vector<cv::Mat>& imgs);
private:
    void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream);
    ICudaEngine* createEngine_s(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize);

};
#endif
