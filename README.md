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

## Build and Run

```shell

git clone https://github.com/RichardoMrMu/yolov5-deepsort-tensorrt.git
cd yolov5-deepsort-tensorrt
mkdir build 
cmake ..
make 
```
if you meet some errors in cmake and make, please see this [artical](https://blog.csdn.net/weixin_42264234/article/details/120152117) or see Attention.

## Model
You need two model, one is yolov5 model, for detection, generating from [tensorrtx](https://github.com/wang-xinyu/tensorrtx). And the other is deepsort model, for tracking. You should generate the model the same way.
### Generate yolov5 model
For yolov5 detection model, I choose yolov5s, and choose `yolov5s.pt->yolov5s.wts->yolov5s.engine`
Note that, used models can get from [yolov5](https://github.com/ultralytics/yolov5) and [deepsort](https://github.com/ZQPei/deep_sort_pytorch), and if you need to use your own model, you can follow the [Custom model,waiting for complete]().
You can see [tensorrtx official readme](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5)

1. Get yolov5 repository


Note that, here uses the official pertained model.And I use yolov5-5, v5.0. So if you train your own model, please be sure your yolov5 code is v5.0.

```shell
git clone -b v5.0 https://github.com/ultralytics/yolov5.git
cd yolov5
mkdir weights
cd weights
// download https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt

```

2. Get tensorrtx.

```shell
git clone https://github.com/wang-xinyu/tensorrtx
```

3. Get xxx.wst model

```shell
cp tensorrtx/gen_wts.py yolov5/
cd yolov5 
python3 gen_wts.py -w ./weights/yolov5s.pt -o ./weights/yolov5s.wts
// a file 'yolov5s.wts' will be generated.
```
You can get yolov5s.wts model in `yolov5/weights/`

4. Build tensorrtx/yolov5 and get tensorrt engine

```shell 
cd tensorrtx/yolov5
// update CLASS_NUM in yololayer.h if your model is trained on custom dataset
mkdir build
cd build
cp {ultralytics}/yolov5/yolov5s.wts {tensorrtx}/yolov5/build
cmake ..
make
// yolov5s
sudo ./yolov5 -s yolov5s.wts yolov5s.engine s
// test your engine file
sudo ./yolov5 -d yolov5s.engine ../samples
```
Then you get the yolov5s.engine, and you can put `yolov5s.engine` in My project. For example

```shell
cd {yolov5-deepsort-tensorrt}
mkdir resources
cp {tensorrtx}/yolov5/build/yolov5s.engine {yolov5-deepsort-tensorrt}/resources
```

5. Get deepsort engine file
You can get deepsort pretrained model in this [drive url](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6)
and ckpt.t7 is ok.
```shell
git clone https://github.com/RichardoMrMu/deepsort-tensorrt.git
// 根据github的说明
cp {deepsort-tensorrt}/exportOnnx.py {deep_sort_pytorch}/
python3 exportOnnx.py
mv {deep_sort_pytorch}/deepsort.onnx {deepsort-tensorrt}/resources
cd {deepsort-tensorrt}
mkdir build
cd build
cmake ..
make 
.onnx2engine ../resources/deepsort.onnx ../resources/deepsort.engine
// test
./demo ../resource/deepsort.engine ../resources/track.txt
```
After all 5 step, you can get the yolov5s.engine and deepsort.engine.

You may face some problems in getting yolov5s.engine and deepsort.engine, you can upload your issue in github or [csdn artical](https://blog.csdn.net/weixin_42264234/article/details/120152117).




