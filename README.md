# fast-yolo-deploy
**Simple and Fast C++ deployment of many YOLO object detection models**  
  
![C++](https://img.shields.io/badge/language-C%2B%2B-f34b7d?logo=c%2B%2B)
![OpenCV](https://img.shields.io/badge/library-OpenCV-5c3ee8?logo=opencv)
![TensorRT](https://img.shields.io/badge/library-TensorRT-76b900?logo=tensorrt)
![ncnn](https://img.shields.io/badge/library-ncnn-616161?logo=ncnn)

# TensorRT Deployment
## Requirements
### Libraries
- TensorRT 8.5
- OpenCV 4.10
### Files
- yolo onnx file OR trt engine file
- Input image OR video OR camera

### Get trt engine file
If exists, you can skip this step.
```bash
path/to/trtexec --onnx=yolo.onnx --fp16 --saveEngine=yolo.engine
```

## Usage
```bash
cd ./src/trt

# build
mkdir build
cd build
cmake ..

# make
make yolo
mv yolo ../

# run
./yolo [inference_mode] [engine_file] [image_file_OR_video_file_OR_camera_index]
```

# ncnn Deployment
**TODO**