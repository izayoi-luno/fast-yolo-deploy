#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

int camera_idx_0 = 0;
int camera_idx_1 = 1;
int FRAME_W = 640;  // This constant is only used for camera property setting
int FRAME_H = 480;
float FOCAL_LEN = 1e-3f;
float BALL_SIZE = 1.8e-1f;
float SENSOR_SIZE_W = 5.37e-3f;
float SENSOR_SIZE_H = 4.04e-3f;
int CONSOLUTION_W = 1920;
int CONSOLUTION_H = 1080;
float PIXEL_SIZE_W = SENSOR_SIZE_W / CONSOLUTION_W;
float PIXEL_SIZE_H = SENSOR_SIZE_H / CONSOLUTION_H;
bool multi_camera = false;
bool bottom = false;
bool need_rotate = false;

void projection(cv::Mat& img, std::vector<std::vector<int>> &objects, std::vector<std::vector<float>> &coordinates, bool bottom) {
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
        positions[i * 2] = (objects[i][0] - frame_center[0]) * PIXEL_SIZE_W;
        positions[i * 2 + 1] = (objects[i][1] - frame_center[1]) * PIXEL_SIZE_H;
        on_sensor_sizes[i] = (objects[i][2] * PIXEL_SIZE_W + objects[i][3] * PIXEL_SIZE_H) / 2;

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

void getMask(cv::Mat &img, std::vector<std::vector<int>> &positions) {
    cv::Mat img_hsv;
    cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);

    cv::Scalar lower_hsv(19, 76, 188);
    cv::Scalar upper_hsv(32, 255, 255);
    cv::Mat mask;
    cv::inRange(img_hsv, lower_hsv, upper_hsv, mask);

    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
    cv::Mat cleaned_mask;
    cv::morphologyEx(mask, cleaned_mask, cv::MORPH_OPEN, kernel);
    cv::Mat rounded_mask;
    cv::morphologyEx(cleaned_mask, rounded_mask, cv::MORPH_CLOSE, kernel);
    cv::Mat dilated_mask;
    cv::dilate(rounded_mask, dilated_mask, kernel, cv::Point(-1, -1), 2);

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(dilated_mask, labels, stats, centroids, 8);
    int min_size = 150;
    cv::Mat large_components_mask = cv::Mat::zeros(dilated_mask.size(), CV_8UC1);
    for (int i = 1; i < num_labels; ++i) {  // index from 1, 0 is the background
        cv::Rect bbox = cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT),
                                 stats.at<int>(i, cv::CC_STAT_TOP),
                                 stats.at<int>(i, cv::CC_STAT_WIDTH),
                                 stats.at<int>(i, cv::CC_STAT_HEIGHT));
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        cv::Point2f centroid = cv::Point2f(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
        if (area > min_size) {
            // Mark the connected domains whose area is greater than the threshold as white
            large_components_mask.setTo(255, labels == i);
            // Mark the center point and bounding box of the connected domain on the original image
            cv::circle(img, centroid, 5, cv::Scalar(0, 255, 0), -1);  // Draw the center point
            cv::rectangle(img, bbox, cv::Scalar(255, 0, 0), 2);  // Draw the bounding box
            std::vector<int> pos = {static_cast<int>(centroid.x), static_cast<int>(centroid.y),
                                    static_cast<int>(bbox.width), static_cast<int>(bbox.height)};
            positions.push_back(pos);
        }
    }

    cv::Mat result;
    cv::bitwise_and(img, img, result, large_components_mask);

    // cv::namedWindow("HSV Threshold", cv::WINDOW_NORMAL);
    // cv::imshow("Original Image", img);
    // cv::imshow("HSV Mask", large_components_mask);
    // cv::imshow("Segmented Image", result);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
}

void testSingleImg(std::string &img_file) {
    cv::Mat img = cv::imread(img_file);
    if (need_rotate) cv::rotate(img, img, cv::ROTATE_180);
    std::vector<std::vector<int>> positions;
    std::vector<std::vector<float>> coordinates;
    getMask(img, positions);
    projection(img, positions, coordinates, bottom);
    for (size_t i = 0; i < positions.size(); ++i) {
        std::cout << "cx: " << positions[i][0] << "    cy: " << positions[i][1] << 
        "    w: " << positions[i][2] << "    h: " << positions[i][3] << std::endl;
        std::cout << "x: " << coordinates[i][0] << "    y: " << coordinates[i][1] << 
        "    z: " << coordinates[i][2] << std::endl;
    }
    cv::namedWindow("HSV Threshold", cv::WINDOW_NORMAL);
    cv::imshow("Original Image", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
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

void testVideo(std::string &video_path) {
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
        cap_0.open(camera_idx_0, cv::CAP_V4L2);  // Use Video4Linux2 backend on Linux
        if (!cap_0.isOpened()) {
            std::cerr << "Error: Could not open camera." << std::endl;
            return;
        }
        std::cout << "Supported Backend: " << cap_0.getBackendName() << std::endl;
        if (multi_camera) {
            cap_1.open(camera_idx_1, cv::CAP_V4L2);
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

    cv::namedWindow("Video_0", cv::WINDOW_NORMAL);
    cv::namedWindow("Video_1", cv::WINDOW_NORMAL);

    cv::Mat frame_0;
    cv::Mat frame_1;
    while (true) {
        cap_0 >> frame_0;
        if (multi_camera) cap_1 >> frame_1;
        
        if (frame_0.empty() || (multi_camera && frame_1.empty())) {
            std::cerr << "Error: Captured frame is empty. Exiting..." << std::endl;
            break;
        }

        std::vector<std::vector<int>> positions_0;
        std::vector<std::vector<int>> positions_1;
        std::vector<std::vector<float>> coordinates_0;
        std::vector<std::vector<float>> coordinates_1;
        getMask(frame_0, positions_0);
        projection(frame_0, positions_0, coordinates_0, bottom);
        if (multi_camera) {
            getMask(frame_1, positions_1);
            projection(frame_1, positions_1, coordinates_1, !bottom);
        }

        for (size_t i = 0; i < positions_0.size(); ++i) {
            std::cout << "cx: " << positions_0[i][0] << "    cy: " << positions_0[i][1] << 
            "    w: " << positions_0[i][2] << "    h: " << positions_0[i][3] << std::endl;
            std::cout << "x: " << coordinates_0[i][0] << "    y: " << coordinates_0[i][1] << 
            "    z: " << coordinates_0[i][2] << std::endl;
        }
        if (multi_camera) {
            for (size_t i = 0; i < positions_1.size(); ++i) {
                std::cout << "cx: " << positions_1[i][0] << "    cy: " << positions_1[i][1] << 
                "    w: " << positions_1[i][2] << "    h: " << positions_1[i][3] << std::endl;
                std::cout << "x: " << coordinates_1[i][0] << "    y: " << coordinates_1[i][1] << 
                "    z: " << coordinates_1[i][2] << std::endl;
            }
        }

        cv::namedWindow("HSV Threshold", cv::WINDOW_NORMAL);
        cv::imshow("Video_0", frame_0);
        if (multi_camera) cv::imshow("Video_1", frame_1);
        if (cv::waitKey(1) == 'q') break;
        cv::destroyAllWindows();
    }
    cap_0.release();
    out_0.release();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Incorrect arguments" << std::endl;
        std::cout << "Basic usage: ./segment [inference mode] [image file OR video file OR camera index]" << std::endl;
        return -1;
    }
    int inference_mode = atoi(argv[1]);
    std::string img_file;
    std::string video_file;
    if (inference_mode == 0) img_file = std::string(argv[2]);
    else if (inference_mode == 1) video_file = std::string(argv[2]);
    else if (inference_mode == 2) {
        camera_idx_0 = atoi(argv[2]);
        if (argc == 4) camera_idx_1 = atoi(argv[3]);
    }

    need_rotate = true;

    if (inference_mode == 0) testSingleImg(img_file);
    else if (inference_mode == 1 || inference_mode == 2) testVideo(video_file);
    else {
        std::cout << "Inference mode can only be 0, 1, and 2" << std::endl;
        std::cout << "0: single image inference\n" << "1: video inference\n" << "2: camera inference" << std::endl;
    }
    return 0;
}