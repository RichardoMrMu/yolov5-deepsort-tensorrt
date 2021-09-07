#include "deepsortenginegenerator.h"
#include "assert.h"
#include <memory.h> 
#include <fstream>

DeepSortEngineGenerator::DeepSortEngineGenerator(ILogger* gLogger) {
    this->gLogger = gLogger;
}

DeepSortEngineGenerator::~DeepSortEngineGenerator() {

}

void DeepSortEngineGenerator::setFP16(bool state) {
    this->useFP16 = state;
}

void DeepSortEngineGenerator::createEngine(std::string onnxPath, std::string enginePath) {
    // Load onnx model
    auto builder = createInferBuilder(*gLogger);
    assert(builder != nullptr);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    assert(network != nullptr);
    auto config = builder->createBuilderConfig();
    assert(config != nullptr);

    auto profile = builder->createOptimizationProfile();
    Dims dims = Dims4{1, 3, IMG_HEIGHT, IMG_WIDTH};
    profile->setDimensions(INPUT_NAME.c_str(),
                OptProfileSelector::kMIN, Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
    profile->setDimensions(INPUT_NAME.c_str(),
                OptProfileSelector::kOPT, Dims4{MAX_BATCH_SIZE, dims.d[1], dims.d[2], dims.d[3]});
    profile->setDimensions(INPUT_NAME.c_str(),
                OptProfileSelector::kMAX, Dims4{MAX_BATCH_SIZE, dims.d[1], dims.d[2], dims.d[3]});
    config->addOptimizationProfile(profile);

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, *gLogger);
    assert(parser != nullptr);
    auto parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
    assert(parsed);
    if (useFP16) config->setFlag(BuilderFlag::kFP16);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // Serialize model and save engine
    IHostMemory* modelStream = engine->serialize();
    std::string serializeStr;
    std::ofstream serializeOutputStream;
    serializeStr.resize(modelStream->size());
    memcpy((void*)serializeStr.data(), modelStream->data(), modelStream->size());
    serializeOutputStream.open(enginePath);
    serializeOutputStream << serializeStr;
    serializeOutputStream.close();
}