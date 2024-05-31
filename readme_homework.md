# VOC数据集目标检测实验

1. mmdetection准备

    按照官网https://github.com/open-mmlab/mmdetection教程下载并配置相应环境

2. VOC数据集准备

    a.进入mmdetection： cd mmdetection

    b.新建数据集文件夹： mkdir datasets

    c.数据集官网下载VOC2007和VOC2012的训练集和VOC2017的测试集到datasets文件夹并解压

3. 训练

    yolov3: python tools/train.py configs/yolo/yolov3_d53_8xb8-ms-608-273e_voc.py

    faster-rcnn: python tools/train.py configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py

    faster-rcnn改进: python tools/train.py configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712-ms-aug.py

4. 训练结果

    实验报告中对应训练权重和日志已保存到mmdetection/work_dirs

    查看训练loss和mAP曲线命令：

        python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/{配置文件名称}/20240514_143328/vis_data/20240514_143328.json --keys {loss或者mAP} --out {图片名称}.jpg

5. 单张图片推理

    替换mytools/infer.py 中的config_path， checkpoint， test_img变量值

    python mytools/infer.py 