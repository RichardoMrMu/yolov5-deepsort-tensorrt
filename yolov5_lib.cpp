//yolov5_lib.cpp 
 
#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include "yolov5_lib.h"
#include "cuda_utils.h"
#include "utils.h"
#include "datatype.h"
#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
 
// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;
 
 
static void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    //cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream);
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    //cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}
 
 
typedef struct 
{
 
    float *data;
    float *prob;
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *exe_context;
    void* buffers[2];
    cudaStream_t cuda_stream;
    int inputIndex;
    int outputIndex;
    char result_json_str[16384];
 
}Yolov5TRTContext;

typedef struct{
	int class_id;
	int x1;
	int y1;
	int x2; 
	int y2;
	float conf;
	
}DeepsortContext;

void * yolov5_trt_create(const char * engine_name)
{
    size_t size = 0;
    char *trtModelStream = NULL;
    Yolov5TRTContext * trt_ctx = NULL;
 
    trt_ctx = new Yolov5TRTContext();
 
    std::ifstream file(engine_name, std::ios::binary);
    printf("yolov5_trt_create  ... \n");
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }else
        return NULL;
 
    trt_ctx->data = new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    trt_ctx->prob = new float[BATCH_SIZE * OUTPUT_SIZE];
    trt_ctx->runtime = createInferRuntime(gLogger);
    assert(trt_ctx->runtime != nullptr);
 
    printf("yolov5_trt_create  cuda engine... \n");
    trt_ctx->engine = trt_ctx->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(trt_ctx->engine != nullptr);
    trt_ctx->exe_context = trt_ctx->engine->createExecutionContext();
 
 
    delete[] trtModelStream;
    assert(trt_ctx->engine->getNbBindings() == 2);
 
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    trt_ctx->inputIndex = trt_ctx->engine->getBindingIndex(INPUT_BLOB_NAME);
    trt_ctx->outputIndex = trt_ctx->engine->getBindingIndex(OUTPUT_BLOB_NAME);
 
    assert(trt_ctx->inputIndex == 0);
    assert(trt_ctx->outputIndex == 1);
    // Create GPU buffers on device
 
    printf("yolov5_trt_create  buffer ... \n");
    CUDA_CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    //cudaMalloc(&trt_ctx->buffers[trt_ctx->inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float));
    CUDA_CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    //cudaMalloc(&trt_ctx->buffers[trt_ctx->outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    // Create stream
 
    printf("yolov5_trt_create  stream ... \n");
    CUDA_CHECK(cudaStreamCreate(&trt_ctx->cuda_stream));
    //cudaStreamCreate(&trt_ctx->cuda_stream);
    printf("yolov5_trt_create  done ... \n");
    return (void *)trt_ctx;
 
 
}
 
 
int yolov5_trt_detect(void *h, cv::Mat &img, float threshold,std::vector<DetectBox>& det)
{
    Yolov5TRTContext *trt_ctx;
    int i;
    int delay_preprocess;
    int delay_infer;
 
    trt_ctx = (Yolov5TRTContext *)h;
 
 
    trt_ctx->result_json_str[0] = 0;
 	// whether det is empty , if not, empty det
 	if (!det.empty()) det.clear();
    if (img.empty()) return 0;
 
    auto start0 = std::chrono::system_clock::now();
 
    //printf("yolov5_trt_detect start preprocess img \n");
    cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);
 	//std::cout<<"after preprocess_img pr_img size:"<<pr_img.cols<<" "<<pr_img.rows<<std::endl;
 	//std::cout<<"after preprocess_img frame size:"<<img.cols<<" "<<img.rows<<std::endl;
 
 
    //printf("yolov5_trt_detect start convert img to float\n");
    // letterbox BGR to RGB
    i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            trt_ctx->data[i] = (float)uc_pixel[2] / 255.0;
            trt_ctx->data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            trt_ctx->data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
    auto end0 = std::chrono::system_clock::now();
 
    delay_preprocess =  std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0).count();
 
    // Run inference
    //printf("yolov5_trt_detect start do inference\n");
    auto start = std::chrono::system_clock::now();
    doInference(*trt_ctx->exe_context, trt_ctx->cuda_stream, trt_ctx->buffers, trt_ctx->data, trt_ctx->prob, BATCH_SIZE);
 
    auto end = std::chrono::system_clock::now();
    delay_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
 
    std::cout <<"delay_proress:" << delay_preprocess << "ms, " << "delay_infer:" << delay_infer << "ms" << std::endl;
 
    //printf("yolov5_trt_detect start do process infer result \n");
 
    int fcount = 1;
    int str_len;
    std::vector<std::vector<Yolo::Detection>> batch_res(1);
    auto& res = batch_res[0];
    nms(res, &trt_ctx->prob[0], threshold, NMS_THRESH);

 
    i = 0;
    for(i = 0 ; i < res.size(); i++){
        int x1, y1, x2, y2;
        int class_id;
        float conf;
        cv::Rect r = get_rect(img, res[i].bbox);
 
        DetectBox dd(r.x,r.y,r.x + r.width,r.y + r.height,(float)res[i].conf,(int)res[i].class_id);
        det.push_back(dd);
    }
    return 1;
}
 
 
void yolov5_trt_destroy(void *h)
{
    Yolov5TRTContext *trt_ctx;
 
    trt_ctx = (Yolov5TRTContext *)h;
 
    // Release stream and buffers
    cudaStreamDestroy(trt_ctx->cuda_stream);
    CUDA_CHECK(cudaFree(trt_ctx->buffers[trt_ctx->inputIndex]));
    //cudaFree(trt_ctx->buffers[trt_ctx->inputIndex]);
    CUDA_CHECK(cudaFree(trt_ctx->buffers[trt_ctx->outputIndex]));
    //cudaFree(trt_ctx->buffers[trt_ctx->outputIndex])
    // Destroy the engine
    trt_ctx->exe_context->destroy();
    trt_ctx->engine->destroy();
    trt_ctx->runtime->destroy();
 
    delete trt_ctx->data;
    delete trt_ctx->prob;
 
    delete trt_ctx;
 
}

