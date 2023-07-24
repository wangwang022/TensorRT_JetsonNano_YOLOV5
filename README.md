# TensorRT_JetsonNano_YOLOV5
# 介绍

这是一个用于在 Jetson Nano 上部署 TensorRT 目标检测模型的简单封装 API，可用于快速部署 yolov5 模型进行目标检测任务。

# 所需环境

必需库：

tensorrt

pycuda

ctypes

对版本没有强制要求。Jetson Nano 的官方镜像会自带 tensorrt，只需要安装对应的 pycuda 即可。若出现问题可以与我联系。

# 使用方法

```
import ctypes
from detect_trt import  YoLov5TRT
import cv2

PLUGIN_LIBRARY = "libmyplugins.so"  #转换trt模型时同时出现的动态库文件
engine_file_path = "yourtrtmodel.engine" #自己转化成的engine文件
ctypes.CDLL(PLUGIN_LIBRARY)  #加载共享库

yolov5_detector = YoLov5TRT(engine_file_path)  #实例化检测函数的类

pred_box = yolov5_detector.detect(frame)  #返回的数据格式为:[[x1,y1,x2,y2,label,conf],....]  frame为读进来的图片

#将目标进行画框可使用类内函数DrawAndText4detect，输入为(frmae,pred_box)
if pred_box:
  for i in range(len(pred_box)):
    YoLov5TRT.DrawAndText4detect(frmae,pred_box[i])
```

# 其他

1.本 API 非本人原创，仅为简单整理后发出供大家参考

2.trt 模型转化方法后续会出，有问题或需要帮忙可以加 qq:103092794
s
