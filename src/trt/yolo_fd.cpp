#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>

#include "opencv2/opencv.hpp"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "cuda_runtime_api.h"

// Helper function to check CUDA errors
#define CHECK_CUDA(status) { if (status != 0) { std::cerr << "CUDA Error: " << status << std::endl; exit(-1); }}

// configs for model inference
int inference_mode = 0;  // default inference mode is single image inference
int input_batch = 1;
int input_channel = 3;
int input_height = 640;
int input_width = 640;
int output_batch = 1;
int output_dim_1 = 84;
int output_dim_2 = 8400;
float scale_ratio = 1.0f;

// configs for post process
float CONF_THREDS = 0.1f;
float IOU_THREDS = 0.7f;
int NUM_CLASS = 5;
int MAX_DET = 300;
int MAX_TIME_IMG = 0.05;
int MAX_NMS = 30000;
int MAX_WH = 7680;
bool IN_PLACE = true;

// params of camera
int camera_index_0 = 0;
int camera_index_1 = 1;
int FRAME_W = 640;  // This constant is only used for camera property setting
int FRAME_H = 480;  // This constant is only used for camera property setting
int CONSOLUTION_W = 1920;
int CONSOLUTION_H = 1080;
float FOCAL_LEN = 1e-3f;
float BALL_SIZE = 1.8e-1f;
float SENSOR_SIZE_W = 5.37e-3f;
float SENSOR_SIZE_H = 4.04e-3f;
float PIXEL_SIZE_W = SENSOR_SIZE_W / CONSOLUTION_W;
float PIXEL_SIZE_H = SENSOR_SIZE_H / CONSOLUTION_H;
bool is_bottom = false;  // if the camera is down
bool multi_camera = false;  // infer with multiple cameras

// other params for testing model performances
bool need_rotate = false;
bool need_crop = false;  // whether need to crop the image into a square
bool check_inference_preprocess = false;  // whether check the preprocess inside inference
bool check_inference = false;  // whether check the inference result
bool dummy = false;  // whether dummy inference
float dummy_nums = 0.9f;

// classes config
std::unordered_map<std::string, int> classes = {
    {"person", 0}, {"bicycle", 1}, {"car", 2}, {"motorcycle", 3}, {"airplane", 4},
    {"bus", 5}, {"train", 6}, {"truck", 7}, {"boat", 8}, {"traffic light", 9},
    {"fire hydrant", 10}, {"stop sign", 11}, {"parking meter", 12}, {"bench", 13},
    {"bird", 14}, {"cat", 15}, {"dog", 16}, {"horse", 17}, {"sheep", 18},
    {"cow", 19}, {"elephant", 20}, {"bear", 21}, {"zebra", 22}, {"giraffe", 23},
    {"backpack", 24}, {"umbrella", 25}, {"handbag", 26}, {"tie", 27}, {"suitcase", 28},
    {"frisbee", 29}, {"skis", 30}, {"snowboard", 31}, {"sports ball", 32}, {"kite", 33},
    {"baseball bat", 34}, {"baseball glove", 35}, {"skateboard", 36}, {"surfboard", 37},
    {"tennis racket", 38}, {"bottle", 39}, {"wine glass", 40}, {"cup", 41}, {"fork", 42},
    {"knife", 43}, {"spoon", 44}, {"bowl", 45}, {"banana", 46}, {"apple", 47},
    {"sandwich", 48}, {"orange", 49}, {"broccoli", 50}, {"carrot", 51}, {"hot dog", 52},
    {"pizza", 53}, {"donut", 54}, {"cake", 55}, {"chair", 56}, {"couch", 57},
    {"potted plant", 58}, {"bed", 59}, {"dining table", 60}, {"toilet", 61}, {"tv", 62},
    {"laptop", 63}, {"mouse", 64}, {"remote", 65}, {"keyboard", 66}, {"cell phone", 67},
    {"microwave", 68}, {"oven", 69}, {"toaster", 70}, {"sink", 71}, {"refrigerator", 72},
    {"book", 73}, {"clock", 74}, {"vase", 75}, {"scissors", 76}, {"teddy bear", 77},
    {"hair drier", 78}, {"toothbrush", 79}
};
std::vector<std::string> focused_classes;  // focused on some classes
std::unordered_set<int> focused_labels;
bool focus = false;

struct Object {
    int label{};
    float prob{};
    cv::Rect_<float> rect;
};

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // Filter out info-level messages
        if (severity <= Severity::kWARNING)
            std::cerr << msg << std::endl;
    }
} gLogger;

std::vector<char> readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size))
    {
        return buffer;
    }
    throw std::runtime_error("Failed to load file: " + filename);
}

void write_result(float* h_data, std::string &file_name, std::string &flag) {
    int input_ele_num = input_batch * input_channel * input_height * input_width;
    int output_ele_num = output_batch * output_dim_1 * output_dim_2;
    int num;
    if (flag == "preProcess") num = input_ele_num; 
    else if (flag == "inference") num = output_ele_num;
    else {
        std::cerr << "flag error: you can only test preprocess or inference result." << std::endl;
        return;
    }

    std::ofstream outfile(file_name);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }
    float* temp_ptr = static_cast<float*>(h_data);
    for (int i = 1; i <= output_batch * output_dim_1 * output_dim_2; ++i) {
        outfile << std::fixed << std::setprecision(6) << temp_ptr[i - 1] << "\t";
        if (!(i % (output_dim_2))) outfile << std::endl;  // Change line
    }
    outfile.close();
}

void test_dummy_data(float* h_data, float dummy_data) {
    int input_ele_num = input_batch * input_channel * input_height * input_width;
    float* dummy_data_ptr = static_cast<float*>(h_data);
    for (int i = 0; i < input_ele_num; ++i) {
        dummy_data_ptr[i] = dummy_data;
    }
}

void preProcess(cv::Mat& img_orig, cv::Mat& img_input, float* h_input_data) {
    // Rotate the img if needed
    if (need_rotate) {
        cv::rotate(img_orig, img_orig, cv::ROTATE_180);
    }
    // Crop the img into a squre if needed
    if (need_crop) {
        if (img_orig.cols > img_orig.rows) {
            int crop = (img_orig.cols - img_orig.rows) / 2;
            cv::Rect crop_rect(crop, 0, img_orig.cols - 2 * crop, img_orig.rows);
            img_orig = img_orig(crop_rect);
        } else if (img_orig.cols < img_orig.rows) {
            int crop = (img_orig.rows - img_orig.cols) / 2;
            cv::Rect crop_rect(0, crop, img_orig.cols, img_orig.rows - 2 * crop);
            img_orig = img_orig(crop_rect);
        }
    }

    // Scale ratio (new / old)
    float r = std::min((float)input_height / (float)img_orig.rows, (float)input_width / (float)img_orig.cols);  
    std::vector<int> new_unpad = {(int)std::round(img_orig.cols * r), (int)std::round(img_orig.rows * r)};

    // Resize
    cv::Mat temp;
    if (img_orig.cols != new_unpad[0] || img_orig.rows != new_unpad[1]) {
        cv::resize(img_orig, temp, cv::Size(new_unpad[0], new_unpad[1]));
    } else {
        temp = img_orig.clone();
    }

    // Add border
    float dw = (input_width - new_unpad[0]) / 2.0f;
    float dh = (input_height - new_unpad[1]) / 2.0f;  // wh padding
    int top = std::round(dh - 0.1f);
    int bottom = std::round(dh + 0.1f);
    int left = std::round(dw - 0.1f);
    int right = std::round(dw + 0.1f);
    cv::Scalar borderColor(114, 114, 114);
    cv::copyMakeBorder(temp, temp, top, bottom, left, right, cv::BORDER_CONSTANT, borderColor);
    cv::cvtColor(temp, img_input, cv::COLOR_BGR2RGB);  // BGR -> RGB
    
    // HWC->CHW and Normalization
    int input_img_area = img_input.cols * img_input.rows;
    unsigned char* input_img_ptr = img_input.data;
    float* h_input_img_r_ptr = h_input_data + input_img_area * 0;
    float* h_input_img_g_ptr = h_input_data + input_img_area * 1;
    float* h_input_img_b_ptr = h_input_data + input_img_area * 2;
    for (int i = 0; i < input_img_area; ++i) {
        *h_input_img_r_ptr++ = input_img_ptr[0] / 255.0f;
        *h_input_img_g_ptr++ = input_img_ptr[1] / 255.0f;
        *h_input_img_b_ptr++ = input_img_ptr[2] / 255.0f;
        input_img_ptr += 3;
    }

    scale_ratio =  1.0f / r;
}

void postProcess(cv::Mat &img_orig, std::vector<Object> &objects, float* h_output_data) {
    float* h_output_data_ptr = h_output_data;
    int num_classes = output_dim_1 - 4;
    int num_anchors = output_dim_2;

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    cv::Mat output = cv::Mat(output_dim_1, output_dim_2, CV_32F, h_output_data_ptr);
    output = output.t();

    for(int i = 0; i < num_anchors; ++i) {
        auto row_ptr = output.row(i).ptr<float>();
        auto bbox_ptr = row_ptr;
        auto score_ptr = row_ptr + 4;
        auto max_score_ptr = std::max_element(score_ptr, score_ptr + num_classes);
        float score = *max_score_ptr;
        if (score > CONF_THREDS) {
            int label = max_score_ptr - score_ptr;
            if (focus) {
                if (focused_labels.find(label) == focused_labels.end()) continue;  // not the focused classes
            }

            float x = *bbox_ptr++;
            float y = *bbox_ptr++;
            float w = *bbox_ptr++;
            float h = *bbox_ptr;

            float x0 = std::clamp((x - 0.5f * w) * scale_ratio, 0.f, (float)img_orig.cols);
            float y0 = std::clamp((y - 0.5f * h) * scale_ratio, 0.f, (float)img_orig.rows);
            float x1 = std::clamp((x + 0.5f * w) * scale_ratio, 0.f, (float)img_orig.cols);
            float y1 = std::clamp((y + 0.5f * h) * scale_ratio, 0.f, (float)img_orig.rows);

            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
        }
    }

    // cv::dnn::NMSBoxesBatched(bboxes, scores, labels, CONF_THREDS, IOU_THREDS, indices);
    cv::dnn::NMSBoxes(bboxes, scores, CONF_THREDS, IOU_THREDS, indices);

    int count = 0;
    for (auto &idx : indices) {
        if (count >= MAX_DET) {
            break;
        }
        Object obj{};
        obj.label = labels[idx];
        obj.prob = scores[idx];
        obj.rect = bboxes[idx];
        objects.push_back(obj);        
        ++count;
    }

}

void projection(cv::Mat& img, std::vector<Object> &objects, std::vector<std::vector<float>> &coordinates, bool bottom) {
    int box_nums = static_cast<int>(objects.size());

    float frame_center[2];
    frame_center[0] = static_cast<float>(img.cols / 2);
    frame_center[1] = static_cast<float>(img.rows / 2);

    float positions[box_nums * 2];  // Obtain object relative positions related to centers (physical position)
    float on_sensor_sizes[box_nums];  // Obtain object sizes on camera sensor (physical position)
    float dist[box_nums];  // Calculate distance to object plane
    float real_positions_x[box_nums];  // Calculate real positions (x)
    float real_positions_y[box_nums];  // // Calculate real positions (y)

    coordinates.resize(box_nums, std::vector<float>(3));

    for (int i = 0; i < box_nums; ++i) {
        positions[i * 2] = (objects[i].rect.x + objects[i].rect.width / 2 - frame_center[0]) * PIXEL_SIZE_W;
        positions[i * 2 + 1] = (objects[i].rect.y + objects[i].rect.height / 2 - frame_center[1]) * PIXEL_SIZE_H;
        on_sensor_sizes[i] = (objects[i].rect.width * PIXEL_SIZE_W + objects[i].rect.height * PIXEL_SIZE_H) / 2;

        dist[i] = FOCAL_LEN * BALL_SIZE / on_sensor_sizes[i];
        real_positions_x[i] = dist[i] * positions[i * 2] / FOCAL_LEN;
        real_positions_y[i] = dist[i] * positions[i * 2 + 1] / FOCAL_LEN;
        
        // Move coordinates to the center of machine
        if (bottom) {
            float temp = real_positions_y[i];
            real_positions_y[i] = dist[i];
            dist[i] = -temp;
            dist[i] += 0.11;  // 0.11m is the origin point from the bottom camera
        } else {
            dist[i] += 0.33;  // 0.33m is the origin point from the fromt camera
        }

        // Convert image coordinates to machine coordinates
        real_positions_x[i] = -real_positions_x[i];
        real_positions_y[i] = -real_positions_y[i];

        // add to results
        coordinates[i][0] = dist[i];  // machine x
        coordinates[i][1] = real_positions_x[i];  // machine y
        coordinates[i][2] = real_positions_y[i];  // machine z
    }

}

void inference(std::string &engine_file, std::string &image_path, std::vector<Object> &objects, std::vector<std::vector<float>> &coordinates) {
    // Load TensorRT engine file
    std::vector<char> trt_model_stream = readFile(engine_file);

    // Create TensorRT runtime
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
    }

    // Deserialize the engine
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trt_model_stream.data(), trt_model_stream.size(), nullptr);
    if (!engine) {
        std::cerr << "Failed to create TensorRT engine" << std::endl;
        runtime->destroy();
        return;
    }

    // Create cuda stream
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Create execution context
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create TensorRT execution context" << std::endl;
        engine->destroy();
        runtime->destroy();
        return;
    }

    // Get binding indices for input and output
    const int input_index = engine->getBindingIndex("images");
    const int output_index = engine->getBindingIndex("output0");

    // Define input and output dimensions
    const nvinfer1::Dims input_dims = engine->getBindingDimensions(input_index);
    const nvinfer1::Dims output_dims = engine->getBindingDimensions(output_index);

    // Allocate memory on host and device for inputs and outputs
    int input_size = input_batch * input_channel * input_height * input_width * sizeof(float);
    int output_size = input_batch * output_dim_1 * output_dim_2 * sizeof(float);

    float* h_input_data = nullptr;
    float* h_output_data = nullptr;
    float* d_input_data = nullptr;
    float* d_output_data = nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_input_data, input_size));
    CHECK_CUDA(cudaMallocHost((void**)&h_output_data, output_size));
    CHECK_CUDA(cudaMalloc((void**)&d_input_data, input_size));
    CHECK_CUDA(cudaMalloc((void**)&d_output_data, output_size));

    // Input data
    cv::Mat image = cv::imread(image_path.c_str());
    if (image.empty()){
        std::cout << "Failed to load image" << std::endl;
        return;
    }
    
    cv::Mat input_image(input_height, input_width, CV_8UC3);
    preProcess(image, input_image, h_input_data);
    
    if (check_inference_preprocess) {
        std::string file_name = "preProcess_result_cpp.txt";
        std::string flag = "preProcess";
        write_result(h_input_data, file_name, flag);
    }
    
    // Test dummy data
    if (dummy) {
        test_dummy_data(h_input_data, dummy_nums);
    }

    // Copy data to device
    CHECK_CUDA(cudaMemcpyAsync(d_input_data, h_input_data, input_size, cudaMemcpyHostToDevice, stream));

    // Set up input bindings
    float* bindings[2];
    bindings[input_index] = d_input_data;
    bindings[output_index] = d_output_data;

    // Run inference
    for(int i=0; i<10; i++){
    	auto start = std::chrono::system_clock::now();
    	bool success = context->enqueueV2((void**)bindings, stream, nullptr);
        std::cout << "Is success: " << success <<std::endl;
    	auto end = std::chrono::system_clock::now();
    	std::cout << "Infer time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Copy output data back to host
    CHECK_CUDA(cudaMemcpyAsync(h_output_data, d_output_data, output_size, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    if (check_inference) {
        std::string file_name = "inference_result_cpp.txt";
        std::string flag = "inference";
        write_result(h_output_data, file_name, flag);
    }

    // Post process
    postProcess(image, objects, h_output_data);

    // Object position projection
    projection(image, objects, coordinates, is_bottom);

    // Clean up
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFreeHost(h_input_data));
    CHECK_CUDA(cudaFreeHost(h_output_data));
    CHECK_CUDA(cudaFree(d_input_data));
    CHECK_CUDA(cudaFree(d_output_data));
    context->destroy();
    engine->destroy();  
    runtime->destroy();

    std::cout << "Inference completed" << std::endl;

}

void cameraSet(cv::VideoCapture &cap, cv::VideoWriter &out) {
    // Set up video properties
    cap.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_H);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FPS, 30);

    // Set video writter 
    std::string output_filename = "output.avi";
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)); 
    out.open(output_filename, fourcc, fps, cv::Size(frame_width, frame_height));

    std::cout << "FourCC: " << cap.get(cv::CAP_PROP_FOURCC) << std::endl;
    std::cout << "FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "Frame Width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "Frame Height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
}

void inferenceVideo(std::string &engine_file, std::string &video_path, std::vector<Object> &objects) {
    std::vector<char> trt_model_stream = readFile(engine_file);
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
    }

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trt_model_stream.data(), trt_model_stream.size(), nullptr);
    if (!engine) {
        std::cerr << "Failed to create TensorRT engine" << std::endl;
        runtime->destroy();
        return;
    }

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create TensorRT execution context" << std::endl;
        engine->destroy();
        runtime->destroy();
        return;
    }

    const int input_index = engine->getBindingIndex("images");
    const int output_index = engine->getBindingIndex("output0");

    const nvinfer1::Dims input_dims = engine->getBindingDimensions(input_index);
    const nvinfer1::Dims output_dims = engine->getBindingDimensions(output_index);

    if (multi_camera) input_batch = 2;
    else input_batch = 1;
    int input_size = input_batch * input_channel * input_height * input_width * sizeof(float);
    int output_size = input_batch * output_dim_1 * output_dim_2 * sizeof(float);

    float* h_input_data = nullptr;
    float* h_output_data = nullptr;
    float* d_input_data = nullptr;
    float* d_output_data = nullptr;
    CHECK_CUDA(cudaMallocHost((void**)&h_input_data, input_size));
    CHECK_CUDA(cudaMallocHost((void**)&h_output_data, output_size));
    CHECK_CUDA(cudaMalloc((void**)&d_input_data, input_size));
    CHECK_CUDA(cudaMalloc((void**)&d_output_data, output_size));

    // Set up video capture
    cv::VideoCapture cap_0;
    cv::VideoCapture cap_1;
    cv::VideoWriter out_0;
    cv::VideoWriter out_1;
    if (!video_path.empty()) {
        cap_0.open(video_path);
        if (!cap_0.isOpened()) {
            std::cerr << "Error: Could not open video file." << std::endl;
            return;
        }
    } else {
        cap_0.open(camera_index_0, cv::CAP_V4L2);  // Use Video4Linux2 backend on Linux
        if (!cap_0.isOpened()) {
            std::cerr << "Error: Could not open camera." << std::endl;
            return;
        }
        std::cout << "Supported Backend: " << cap_0.getBackendName() << std::endl;
        if (multi_camera) {
            cap_1.open(camera_index_1, cv::CAP_V4L2);
            if (!cap_1.isOpened()) {
                std::cerr << "Error: Could not open camera." << std::endl;
                return;
            }
        } else {
            cap_1.release();
            out_1.release();
        }
    }
    cameraSet(cap_0, out_0);
    if (multi_camera) cameraSet(cap_1, out_1);

    cv::Mat input_frame_0(input_height, input_width, CV_8UC3);
    cv::Mat input_frame_1(input_height, input_width, CV_8UC3);
    // set buffer size
    cap_0.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap_1.set(cv::CAP_PROP_BUFFERSIZE, 1);

    bool exposureSet_0 = cap_0.set(cv::CAP_PROP_EXPOSURE, -6);
    bool exposureSet_1 = cap_1.set(cv::CAP_PROP_EXPOSURE, -6);
    std::cout << exposureSet_0 << " " << exposureSet_0 << std::endl;

    cv::namedWindow("Video_0", cv::WINDOW_NORMAL);
    cv::namedWindow("Video_1", cv::WINDOW_NORMAL);

    cv::Mat frame_0;
    cv::Mat frame_1;
    while (true) {
    	auto start1 = std::chrono::system_clock::now();
        bool success_0 = cap_0.read(frame_0);
        bool success_1 = true;
        auto start2 = std::chrono::system_clock::now();
        std::cout << "Reading time: " << std::chrono::duration_cast<std::chrono::milliseconds>(start2 - start1).count() << "ms" << std::endl;
        
        if (multi_camera) success_1 = cap_1.read(frame_1);
        if (success_0 && success_1) {
            auto start = std::chrono::system_clock::now();

            preProcess(frame_0, input_frame_0, h_input_data);
            if (multi_camera) preProcess(frame_1, input_frame_1, h_input_data + input_channel * input_height * input_width);
            CHECK_CUDA(cudaMemcpyAsync(d_input_data, h_input_data, input_size, cudaMemcpyHostToDevice, stream));


            float* bindings[2];
            bindings[input_index] = d_input_data;
            bindings[output_index] = d_output_data;

	    // start of gpu infer
            cudaEvent_t start_, stop_;
            CHECK_CUDA(cudaEventCreate(&start_));
            CHECK_CUDA(cudaEventCreate(&stop_));
            CHECK_CUDA(cudaEventRecord(start_));
            cudaEventQuery(start_);
            bool success = context->enqueueV2((void**)bindings, stream, nullptr);
	    // end of gpu infer
            CHECK_CUDA(cudaEventRecord(stop_));
            CHECK_CUDA(cudaEventSynchronize(stop_));
            float elapsed_time;
            CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start_, stop_));
            printf("GPU inference Time = %g ms.\n", elapsed_time);
            
            CHECK_CUDA(cudaMemcpyAsync(h_output_data, d_output_data, output_size, cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            
            std::vector<Object> objects_0;
            std::vector<Object> objects_1;
            postProcess(frame_0, objects_0, h_output_data);
            if (multi_camera) postProcess(frame_1, objects_1, h_output_data + output_dim_1 * output_dim_2);
            
            std::vector<std::vector<float>> coordinates_0;
            std::vector<std::vector<float>> coordinates_1;
            projection(frame_0, objects_0, coordinates_0, is_bottom);
            if (multi_camera) projection(frame_1, objects_1, coordinates_1, !is_bottom);
            auto end = std::chrono::system_clock::now();
            std::cout << "One frame process time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        
            for (int i = 0; i < (int)objects_0.size(); ++i) {
                std::cout << "Position by Camera 0: " << coordinates_0[i][0] << ", " << coordinates_0[i][1] << ", "
                            << coordinates_0[i][2] << "    conf: " << objects_0[i].prob << std::endl; 
            }
            if (multi_camera) {
                for (int i = 0; i < (int)objects_1.size(); ++i) {
                    std::cout << "Position by Camera 1: " << coordinates_1[i][0] << ", " << coordinates_1[i][1] << ", "
                            << coordinates_1[i][2] << "    conf: " << objects_1[i].prob << std::endl; 
                }
            }
            
            cv::imshow("Video_0", frame_0);
            if (multi_camera) cv::imshow("Video_1", frame_1);

            objects_0.clear();
            coordinates_0.clear();
            if (multi_camera) {
                objects_1.clear();
                coordinates_1.clear();
            }


            if (cv::waitKey(1) == 'q') break;
        } else {
            std::cerr << "Error: Could not read frame." << std::endl;
            break; 
        }
    }

    cap_0.release();
    out_0.release();

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFreeHost(h_input_data));
    CHECK_CUDA(cudaFreeHost(h_output_data));
    CHECK_CUDA(cudaFree(d_input_data));
    CHECK_CUDA(cudaFree(d_output_data));
    context->destroy();
    engine->destroy();  
    runtime->destroy();
}

void test_preProcess(std::string &image_file, std::string &file_name, bool show=true) {
    cv::Mat image = cv::imread(image_file);
    if (image.empty()){
        std::cout << "Failed to load image" << std::endl;
        return;
    }

    cv::Mat input_image(input_height, input_width, CV_8UC3);
    int input_ele_num = input_batch * input_channel * input_height * input_width;
    float* h_input_data = nullptr;
    h_input_data = new float[input_ele_num];

    preProcess(image, input_image, h_input_data);
    std::ofstream outfile(file_name);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }
    float* temp_ptr = h_input_data;
    for (int i = 1; i <= input_ele_num; ++i) {
        outfile << *temp_ptr++ << "\t";
        if (!(i % input_width)) outfile << std::endl;
    }
    outfile.close();

    if (show) {
        cv::namedWindow("Image 1", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Image 2", cv::WINDOW_AUTOSIZE);

        cv::imshow("Image 1", image);
        cv::imshow("Image 2", input_image);

        cv::waitKey(0);

        cv::destroyWindow("Image 1");
        cv::destroyWindow("Image 2");
    }
    
    delete[] h_input_data;
    h_input_data = nullptr;
}

void test_postProcess(std::string &image_file, std::string &bin_file) {
    cv::Mat image = cv::imread(image_file);
    cv::Mat input_image(input_height, input_width, CV_8UC3);
    float* h_input_data = new float[input_batch * input_channel * input_height * input_width];
    float* h_ouptut_data = new float[output_batch * output_dim_1 * output_dim_2];

    std::ifstream file(bin_file, std::ios::binary);
    if (!file) {
        std::cerr << "cannot open file" << std::endl;
        return;
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::size_t num_floats = size / sizeof(float);

    if (output_batch * output_dim_1 * output_dim_2 != (int)num_floats) {
        std::cerr << "data size not correct" << std::endl;
        return;
    }

    float* temp = h_ouptut_data;
    if (file.read(reinterpret_cast<char*>(temp), size)) {
        std::cout << "successfully read" << std::endl;
    } else {
        std::cerr << "errors when opening file" << std::endl;
        return;
    }

    preProcess(image, input_image, h_input_data);
    
    std::vector<Object> ret;
    postProcess(image, ret, h_ouptut_data);
    for (int i = 0; i < (int)ret.size(); ++i) {
        std::cout << "result" << i << std::endl;
        std::cout << "label: " << ret[i].label << std::endl;
        std::cout << "score: " << ret[i].prob << std::endl;
        std::cout << "bbox: " << ret[i].rect.x << ", " << ret[i].rect.y << ", "
                    << ret[i].rect.width << ", " << ret[i].rect.height << std::endl;
    }
    
    delete[] h_input_data;
    delete[] h_ouptut_data;
    h_input_data = nullptr;
    h_ouptut_data = nullptr;
}

void test_inference(std::string &engine_file, std::string &image_file) {
    std::vector<Object> ret;
    std::vector<std::vector<float>> coordinates;
    inference(engine_file, image_file, ret, coordinates);
    for (int i = 0; i < (int)ret.size(); ++i) {
        std::cout << "result no." << i << std::endl;
        std::cout << "label: " << ret[i].label << std::endl;
        std::cout << "score: " << ret[i].prob << std::endl;
        std::cout << "bbox: " << ret[i].rect.x << ", " << ret[i].rect.y << ", "
                    << ret[i].rect.width << ", " << ret[i].rect.height << std::endl;
        std::cout << "position: " << coordinates[i][0] << ", " << coordinates[i][1] << ", "
                    << coordinates[i][2] << std::endl; 
    }
}

void test_videoInference(std::string &engine_file, std::string &video_file) {
    std::vector<Object> objects;
    inferenceVideo(engine_file, video_file, objects);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Basic Usage: ./yolo_world_cpp [inference mode] [engine file] [image file OR video file OR camera index]" << std::endl;
        return -1;
    }
    int inference_mode = atoi(argv[1]);
    std::string engine_file = std::string(argv[2]);

    std::string img_file;
    std::string video_file;
    if (inference_mode == 0) img_file = std::string(argv[3]);
    else if (inference_mode == 1) video_file = std::string(argv[3]);
    else if (inference_mode == 2) {
        camera_index_0 = atoi(argv[3]);
        if (argc == 5) camera_index_1 = atoi(argv[4]);
    }
    
    if (img_file.find("down") != std::string::npos) is_bottom = true;

    if (argc == 5 && inference_mode == 2) multi_camera = true;
    need_rotate = false;  // If you need rotate your image or frame
    need_crop = true;  // you can decide whether to crop your original image for better performance
    check_inference_preprocess = false;  // whether to run preProcess debugs
    check_inference = false;  // whetherto run inference debugs
    dummy = false;  // whehter to run dummy inference
    dummy_nums = 0.9f;  // set dummy value

    // you can set the focused classes here
    focused_classes.push_back("frisbee");
    focused_classes.push_back("sports ball");

    focus = !focused_classes.empty(); // if focused on some classes
    if (focus) {
        for (int i  = 0; i < (int)focused_classes.size(); ++i) {
            focused_labels.insert(classes[focused_classes[i]]);
        }
    }

    if (inference_mode == 0) test_inference(engine_file, img_file);
    else if (inference_mode == 1 || inference_mode == 2) test_videoInference(engine_file, video_file);
    else {
        std::cout << "Inference mode can only be 0, 1, and 2" << std::endl;
        std::cout << "0: single image inference\n" << "1: video inference\n" << "2: camera inference" << std::endl;
    }
    return 0;
}
