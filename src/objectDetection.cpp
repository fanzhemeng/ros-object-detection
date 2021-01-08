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
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;


typedef const boost::function< void(const sensor_msgs::ImageConstPtr&)> callback;
const char* sub_topic01 = "/Cam/Image_raw01";
const char* sub_topic02 = "/Cam/Image_raw02";
const char* sub_topic03 = "/Cam/Image_raw03";
const char* pub_topic01 = "/detection01";
const char* pub_topic02 = "/detection02";
const char* pub_topic03 = "/detection03";

ros::Subscriber sub01;
ros::Subscriber sub02;
ros::Subscriber sub03;
ros::Publisher pub01;
ros::Publisher pub02;
ros::Publisher pub03;

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
    ICudaEngine* createEngine_s(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize);
    std::vector<std::vector<Yolo::Detection>> detect(std::vector<cv::Mat>& imgs);
    void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream);
};

// Creat the engine using only the API and not any parser.
ICudaEngine* yolov5sengine::createEngine_s(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("yolov5s.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    // yolov5 backbone
    auto focus0 = focus(network, weightMap, *data, 3, 32, 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), 64, 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), 64, 64, 1, true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 128, 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), 128, 128, 3, true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 256, 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), 256, 256, 3, true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 512, 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), 512, 512, 5, 9, 13, "model.8");

    // yolov5 head
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), 256, 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts11{ DataType::kFLOAT, deval, 256 * 2 * 2 };
    IDeconvolutionLayer* deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), 256, DimsHW{ 2, 2 }, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{ 2, 2 });
    deconv11->setNbGroups(256);
    weightMap["deconv11"] = deconvwts11;

    ITensor* inputTensors12[] = { deconv11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), 512, 256, 1, false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 128, 1, 1, 1, "model.14");

    Weights deconvwts15{ DataType::kFLOAT, deval, 128 * 2 * 2 };
    IDeconvolutionLayer* deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), 128, DimsHW{ 2, 2 }, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{ 2, 2 });
    deconv15->setNbGroups(128);

    ITensor* inputTensors16[] = { deconv15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), 256, 128, 1, false, 1, 0.5, "model.17");
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);

    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 128, 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = bottleneckCSP(network, weightMap, *cat19->getOutput(0), 256, 256, 1, false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);

    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 256, 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = bottleneckCSP(network, weightMap, *cat22->getOutput(0), 512, 512, 1, false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, det0, det1, det2);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));

    }
    return engine;
}

void yolov5sengine::APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = (CREATENET(NET))(maxBatchSize, builder, config, DataType::kFLOAT);
    //ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

yolov5sengine::yolov5sengine() {

    // get classes names from classes.txt
    std::ifstream infile("classes.txt");
    std::string line;
    while (std::getline(infile, line)) { classes.push_back(line); }

    // setup engine
    cudaSetDevice(DEVICE);
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    std::ifstream file(engine_name, std::ios::binary);
    if (! file.good()) {
        std::cout << engine_name << " not found, now try to build it first..." << std::endl;
        
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "Unable to build engine from yolov5s.wts ... Do you have yolov5s.wts in current dir?\nABORT" << std::endl;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
    }
    file.close();
    file.open(engine_name, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    else {
        std::cerr << engine_name << " still not found.\nABORT" << std::endl;
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
    void imageCallback01(const sensor_msgs::ImageConstPtr& msg);
    void imageCallback02(const sensor_msgs::ImageConstPtr& msg);
    void imageCallback03(const sensor_msgs::ImageConstPtr& msg);
    objectDetector();
};

objectDetector::objectDetector() {
    check01 = check02 = check03 = false;
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

        //cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->cvImage01).toImageMsg((this->msgOut01).roi_image);
        //cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->cvImage02).toImageMsg((this->msgOut02).roi_image);
        //cv_bridge::CvImage(std_msgs::Header(), "bgr8", this->cvImage03).toImageMsg((this->msgOut03).roi_image);

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
        pub01.publish(this->msgOut01);
        pub02.publish(this->msgOut02);
        pub03.publish(this->msgOut03);
        
        check01 = check02 = check03 = false;
    }
}



int main(int argc, char** argv)
{
    ros::init(argc, argv, "object_detection_msgs");
    ros::NodeHandle nh;

    pub01 = nh.advertise<waytous_perception_msgs::ObjectArray>(pub_topic01, 1);
    pub02 = nh.advertise<waytous_perception_msgs::ObjectArray>(pub_topic02, 1);
    pub03 = nh.advertise<waytous_perception_msgs::ObjectArray>(pub_topic03, 1);

    objectDetector detector;
    sub01 = nh.subscribe(sub_topic01, 1, &objectDetector::imageCallback01, &detector);
    sub02 = nh.subscribe(sub_topic02, 1, &objectDetector::imageCallback02, &detector);
    sub03 = nh.subscribe(sub_topic03, 1, &objectDetector::imageCallback03, &detector);

    ros::spin();

    return 0;
}
