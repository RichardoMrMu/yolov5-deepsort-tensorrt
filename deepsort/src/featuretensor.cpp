#include "featuretensor.h"
#include <fstream>

using namespace nvinfer1;

#define INPUTSTREAM_SIZE (maxBatchSize*3*imgShape.area())
#define OUTPUTSTREAM_SIZE (maxBatchSize*featureDim)

FeatureTensor::FeatureTensor(const int maxBatchSize, const cv::Size imgShape, const int featureDim, int gpuID, ILogger* gLogger) 
        : maxBatchSize(maxBatchSize), imgShape(imgShape), featureDim(featureDim), 
        inputStreamSize(INPUTSTREAM_SIZE), outputStreamSize(OUTPUTSTREAM_SIZE),
        inputBuffer(new float[inputStreamSize]), outputBuffer(new float[outputStreamSize]),
        inputName("input"), outputName("output") {
    cudaSetDevice(gpuID);
    this->gLogger = gLogger;
    runtime = nullptr;
    engine = nullptr;
    context = nullptr; 

    means[0] = 0.485, means[1] = 0.456, means[2] = 0.406;
    std[0] = 0.229, std[1] = 0.224, std[2] = 0.225;

    initFlag = false;
}

FeatureTensor::~FeatureTensor() {
    delete [] inputBuffer; 
    delete [] outputBuffer;
    if (initFlag) {
        // cudaStreamSynchronize(cudaStream);
        cudaStreamDestroy(cudaStream);
        cudaFree(buffers[inputIndex]);
        cudaFree(buffers[outputIndex]);
    }
}

bool FeatureTensor::getRectsFeature(const cv::Mat& img, DETECTIONS& det) {
    std::vector<cv::Mat> mats;
    for (auto& dbox : det) {
        cv::Rect rect = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
                                 int(dbox.tlwh(2)), int(dbox.tlwh(3)));
        rect.x -= (rect.height * 0.5 - rect.width) * 0.5;
        rect.width = rect.height * 0.5;
        rect.x = (rect.x >= 0 ? rect.x : 0);
        rect.y = (rect.y >= 0 ? rect.y : 0);
        rect.width = (rect.x + rect.width <= img.cols ? rect.width : (img.cols - rect.x));
        rect.height = (rect.y + rect.height <= img.rows ? rect.height : (img.rows - rect.y));
        cv::Mat tempMat = img(rect).clone();
        cv::resize(tempMat, tempMat, imgShape);
        mats.push_back(tempMat);
    }
    doInference(mats);
    // decode output to det
    stream2det(outputBuffer, det);
    return true;
}

bool FeatureTensor::getRectsFeature(DETECTIONS& det) {
    return true;
}

void FeatureTensor::loadEngine(std::string enginePath) {
    // Deserialize model
    runtime = createInferRuntime(*gLogger);
    assert(runtime != nullptr);
    std::ifstream engineStream(enginePath);
    std::string engineCache("");
    while (engineStream.peek() != EOF) {
        std::stringstream buffer;
        buffer << engineStream.rdbuf();
        engineCache.append(buffer.str());
    }
    engineStream.close();
    engine = runtime->deserializeCudaEngine(engineCache.data(), engineCache.size(), nullptr);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    initResource();
} 

void FeatureTensor::loadOnnx(std::string onnxPath) {
    auto builder = createInferBuilder(*gLogger);
    assert(builder != nullptr);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    assert(network != nullptr);
    auto config = builder->createBuilderConfig();
    assert(config != nullptr);

    auto profile = builder->createOptimizationProfile();
    Dims dims = Dims4{1, 3, imgShape.height, imgShape.width};
    profile->setDimensions(inputName.c_str(),
                OptProfileSelector::kMIN, Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
    profile->setDimensions(inputName.c_str(),
                OptProfileSelector::kOPT, Dims4{maxBatchSize, dims.d[1], dims.d[2], dims.d[3]});
    profile->setDimensions(inputName.c_str(),
                OptProfileSelector::kMAX, Dims4{maxBatchSize, dims.d[1], dims.d[2], dims.d[3]});
    config->addOptimizationProfile(profile);

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, *gLogger);
    assert(parser != nullptr);
    auto parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
    assert(parsed);
    config->setMaxWorkspaceSize(1 << 20);
    engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    initResource();
}

int FeatureTensor::getResult(float*& buffer) {
    if (buffer != nullptr)
        delete buffer;
    int curStreamSize = curBatchSize*featureDim;
    buffer = new float[curStreamSize];
    for (int i = 0; i < curStreamSize; ++i) {
        buffer[i] = outputBuffer[i];
    }
    return curStreamSize;
}

void FeatureTensor::doInference(vector<cv::Mat>& imgMats) {
    mat2stream(imgMats, inputBuffer);
    doInference(inputBuffer, outputBuffer);
}

void FeatureTensor::initResource() {
    inputIndex = engine->getBindingIndex(inputName.c_str());
    outputIndex = engine->getBindingIndex(outputName.c_str());
    // Create CUDA stream
    cudaStreamCreate(&cudaStream);
    buffers[inputIndex] = inputBuffer;
    buffers[outputIndex] = outputBuffer;
    
    // Malloc CUDA memory
    cudaMalloc(&buffers[inputIndex], inputStreamSize * sizeof(float));
    cudaMalloc(&buffers[outputIndex], outputStreamSize * sizeof(float));
    
    initFlag = true;
}

void FeatureTensor::doInference(float* inputBuffer, float* outputBuffer) {   
    cudaMemcpyAsync(buffers[inputIndex], inputBuffer, inputStreamSize * sizeof(float), cudaMemcpyHostToDevice, cudaStream);
    Dims4 inputDims{curBatchSize, 3, imgShape.height, imgShape.width};
    context->setBindingDimensions(0, inputDims);
    
    context->enqueueV2(buffers, cudaStream, nullptr);
    cudaMemcpyAsync(outputBuffer, buffers[outputIndex], outputStreamSize * sizeof(float), cudaMemcpyDeviceToHost, cudaStream);
    // cudaStreamSynchronize(cudaStream);
}

void FeatureTensor::mat2stream(vector<cv::Mat>& imgMats, float* stream) {
    int imgArea = imgShape.area();
    curBatchSize = imgMats.size();
    if (curBatchSize > maxBatchSize) {
        std::cout << "[WARNING]::Batch size overflow, input will be truncated!" << std::endl;
        curBatchSize = maxBatchSize;
    }
    for (int batch = 0; batch < curBatchSize; ++batch) {
        cv::Mat tempMat = imgMats[batch];
        int i = 0; 
        for (int row = 0; row < imgShape.height; ++row) {
            uchar* uc_pixel = tempMat.data + row * tempMat.step;
            for (int col = 0; col < imgShape.width; ++col) {
                stream[batch * 3 * imgArea + i] = ((float)uc_pixel[0] / 255.0 - means[0]) / std[0];
                stream[batch * 3 * imgArea + i + imgArea] = ((float)uc_pixel[1] / 255.0 - means[1]) / std[1];
                stream[batch * 3 * imgArea + i + 2 * imgArea] = ((float)uc_pixel[2] / 255.0 - means[2]) / std[2];
                uc_pixel += 3;
                ++i;
            }
        }
    }
}

void FeatureTensor::stream2det(float* stream, DETECTIONS& det) {
    int i = 0;
    for (DETECTION_ROW& dbox : det) {
        for (int j = 0; j < featureDim; ++j)
            dbox.feature[j] = stream[i * featureDim + j];
            // dbox.feature[j] = (float)1.0;
        ++i;
    }
}
