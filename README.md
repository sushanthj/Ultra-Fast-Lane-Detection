# Ultra-Fast-Lane-Detection
PyTorch implementation of the paper "[Ultra Fast Structure-aware Deep Lane Detection](https://arxiv.org/abs/2004.11757)".

**\[July 18, 2022\] Updates: The new version of our method is available [here](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)** --> **NOTE: The Dataloader uses different information for the CULane dataset than this implementation**

![alt text](vis.jpg "vis")

# On-Road Lane Detection Demo
<a href="http://www.youtube.com/watch?feature=player_embedded&v=lnFbAG3GBN4
" target="_blank"><img src="http://img.youtube.com/vi/lnFbAG3GBN4/0.jpg"
alt="Demo" width="240" height="180" border="10" /></a>

# Crop Row Detection Output

![](result_1.png)
![](result_2.png)

# Install and Setup
Please see [INSTALL.md](./INSTALL.md)

# Get started
First of all, please modify `data_root` and `log_path` in your `configs/culane.py` or `configs/tusimple.py` config according to your environment.
- `data_root` is the path of your CULane dataset or Tusimple dataset.
- `log_path` is where tensorboard logs, trained models and code backup are stored. ***It should be placed outside of this project.***


For single gpu training, run
```Shell
python train.py configs/path_to_your_config
```
For multi-gpu training, run
```Shell
sh launch_training.sh
```
or
```Shell
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py configs/path_to_your_config
```
If there is no pretrained torchvision model, multi-gpu training may result in multiple downloading. You can first download the corresponding models manually, and then restart the multi-gpu training.

Since our code has auto backup function which will copy all codes to the `log_path` according to the gitignore, additional temp file might also be copied if it is not filtered by gitignore, which may block the execution if the temp files are large. So you should keep the working directory clean.
***

Besides config style settings, we also support command line style one. You can override a setting like
```Shell
python train.py configs/path_to_your_config --batch_size 8
```
The ```batch_size``` will be set to 8 during training.

***

To visualize the log with tensorboard, run

```Shell
tensorboard --logdir log_path --bind_all
```

# Trained models
We provide two trained Res-18 models on CULane and Tusimple.

|  Dataset | Metric paper | Metric This repo | Avg FPS on GTX 1080Ti |    Model    |
|:--------:|:------------:|:----------------:|:-------------------:|:-----------:|
| Tusimple |     95.87    |       95.82      |         306         | [GoogleDrive](https://drive.google.com/file/d/1WCYyur5ZaWczH15ecmeDowrW30xcLrCn/view?usp=sharing)/[BaiduDrive(code:bghd)](https://pan.baidu.com/s/1Fjm5yVq1JDpGjh4bdgdDLA) |
|  CULane  |     68.4     |       69.7       |         324         | [GoogleDrive](https://drive.google.com/file/d/1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq/view?usp=sharing)/[BaiduDrive(code:w9tw)](https://pan.baidu.com/s/19Ig0TrV8MfmFTyCvbSa4ag) |


# Visualization / Inference

We provide a script to visualize the detection results. Run the following commands to visualize on the testing set of CULane and Tusimple.
```Shell
python demo_v2.py configs/young_soybean_1.py --test_model path_to_culane_18.pth
# or
python demo.py configs/tusimple.py --test_model path_to_tusimple_18.pth
```

Since the testing set of Tusimple is not ordered, the visualized video might look bad and we **do not recommend** doing this.

# Speed
To test the runtime, please run
```Shell
python speed_simple.py  
# this will test the speed with a simple protocol and requires no additional dependencies

python speed_real.py
# this will test the speed with real video or camera input
```
It will loop 100 times and calculate the average runtime and fps in your environment.
```