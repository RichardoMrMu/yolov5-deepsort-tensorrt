# A C++ implementation of Yolov5 and Deepsort in Jetson Xavier nx and Jetson nano
This repository uses yolov5 and deepsort to follow humna heads which can run in Jetson Xavier nx and Jetson nano. 
In Jetson Xavier Nx, it can achieve 10 FPS when images contain heads about 70+(you can try python version, when you use python version, you can find it very slow in Jetson Xavier nx , and Deepsort can cost nearly 1s).

## Requirement
1. Jetson nano or Jetson Xavier nx
2. Jetpack 4.5.1
3. python3 with default(jetson nano or jetson xavier nx has default python3 with tensorrt 7.1.3.0 )
4. tensorrt 7.1.3.0
5. torch 1.8.0
6. torchvision 0.9.0
7. torch2trt 0.3.0


if you have problem in this project, you can see this [artical](https://blog.csdn.net/weixin_42264234/article/details/120152117).
## Speed

Whole process time from read image to finished deepsort (include every img preprocess and postprocess)
and attention!!! the number of deepsort tracking is 70+, not single or 10-20 persons, is 70+. And all results can get in Jetson Xavier nx.
| Backbone        | before TensorRT without tracking |before TensorRT with tracking | TensorRT(detection + tracking) | FPS(detection + tracking) |
| :-------------- | --------------- | ------------------ |------------------------------ | ------------------------- |
| Yolov5s_416      | 100ms           | 0.9s|450ms                          | 1.5 ~ 2                   |
| Yolov5s-640 | 120ms             | 1s|100-150ms                      | 8 ~ 9                     |

------






