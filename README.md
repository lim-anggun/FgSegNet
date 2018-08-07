# FgSegNet : Foreground Segmentation Network

This repository contains source codes and training sets for the following paper:<br /><br />
***["Foreground Segmentation Using Convolutional Neural Networks for Multiscale Feature Encoding"](https://www.sciencedirect.com/science/article/pii/S0167865518303702) by Long Ang LIM and Hacer YALIM KELES*** <br /><br />

The preprint version of the above paper is available at: https://arxiv.org/abs/1801.02225 <br/><br/>

![alt tag](network.png "FgSegNet_M_S Network Architecture")

<br/>
## Citation
If you find FgSegNet useful in your research, please consider citing: <br />
```
@article{LIM2018,
  title = "Foreground Segmentation Using Convolutional Neural Networks for Multiscale Feature Encoding",
  journal = "Pattern Recognition Letters",
  year = "2018",
  issn = "0167-8655",
  doi = "https://doi.org/10.1016/j.patrec.2018.08.002",
  url = "http://www.sciencedirect.com/science/article/pii/S0167865518303702",
  author = "Long Ang Lim and Hacer Yalim Keles",
  keywords = "Foreground segmentation, Background subtraction, Deep learning, Convolutional neural networks, Video surveillance, Pixel classification"
}
```

## Requirements
This work was implemented with the following frameworks:
* Spyder 3.2.x (recommended)
* Python 3.6.3
* Keras 2.0.6
* Tensorflow-gpu 1.1.0

## Usage
Easy to train! Just a single click, go! <br />
1. Clone this repo: ``git clone https://github.com/lim-anggun/FgSegNet.git``

2. Modify the following file:
    * ``<Your PYTHON 3.6>\site-packages\skimage\transform\pyramids.py`` <br/>
          in ```pyramid_reduce``` function, replace the following two lines<br/>
          ```out_rows = math.ceil(rows / float(downscale))``` <br/>
          ```out_cols = math.ceil(cols / float(downscale))``` <br/>
          with <br/>
          ```out_rows = math.floor(rows / float(downscale))```<br/>
          ```out_cols = math.floor(cols / float(downscale))```

3. Download VGG16 weights from [Here](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5) and put it in ```FgSegNet/FgSegNet/``` dir, or it will be downloaded and stored in ```/.keras/models/``` automatically.

4. Download [CDnet2014 dataset](http://changedetection.net), then put it in the following directory structure:<br/>

    Example:

    ```
      FgSegNet/
           FgSegNet/FgSegNet_M_S_CDnet.py
                   /FgSegNet_M_S_SBI.py
                   /FgSegNet_M_S_UCSD.py
                   /FgSegNet_M_S_module.py
           SBI2015_dataset/
           SBI2015_train/
           UCSD_dataset/
           UCSD_train20/
           UCSD_train50/
           FgSegNet_dataset2014/
                     baseline/
                          highway50
                          highway200
                          pedestrians50
                          pedestrians200
                          ...
                      badWeather/
                          skating50
                          skating200
                          ...
                     ...
            CDnet2014_dataset/
                      baseline/
                           highway
                           pedestrians
                           ...
                      badWeather/
                           skating
                           ...
                      ...
    ```

5. There are two methods; i.e. ```FgSegNet_M``` and ```FgSegNet_S```. Choose a method that you want to train by setting ```method_name=='FgSegNet_M' or method_name=='FgSegNet_S'```.

6. Run the codes with **Spyder IDE**. Note that all trained models will be automatically saved (in current working directory) for you.

## Evaluation
### on CDnet2014 dataset

We perform two separated evaluations and report our results on two test splits (```test dev``` & ```test challenge```): <br />
1. We compute our results locally. (on ```test dev``` dataset)
2. We upload our results to [Change Detection 2014 Challenge](http://changedetection.net). (on ```test challenge``` dataset where ground truth values are not shared with the public dataset)<br />
(Both results are reported in our paper. Please refer to it for details)<br />

Compute metrics locally using [CDnet Utilities](http://wordpress-jodoin.dmi.usherb.ca/code/)
#### Note:
- ```test dev```: by considering only the range of the frames that contain the ground truth labels by excluding training frames (50 or 200 frames)
- ```test challenge```: dataset on the server side (http://changedetection.net)
### on SBI2015 dataset

We split 20% for training (denoted by n frames, where n ∈ [2−148]) and 80% for testing.

### on UCSD Background Subtraction dataset

We perform two sets of experiment: first, we split the frames 20% for training (denoted by n frames, where n ∈ [3 − 23]) and 80% for testing, second we split 50% for training (where n ∈ [7 − 56]) and remaining 50% for testing.

## Results
### Results on CDnet2014 dataset
Table below shows overall results across 11 categories obtained from [Change Detection 2014 Challenge](http://changedetection.net).

| Methods  | PWC | F-Measure | Speed (320x240, batch-size=1) on NVIDIA GTX 970 GPU |
| ------------- | ------------- | ------------- | ------------- |
| FgSegNet_M  | 0.0559 | 0.9770 | 18fps |
| FgSegNet_S  | 0.0461 | 0.9804 | 21fps |

### Results on SBI2015 dataset
Table below shows overall *test results* across 14 video sequences.

| Methods  | PWC | F-Measure |
| ------------- | ------------- | ------------- |
| FgSegNet_M  | 0.9431 | 0.9794 |
| FgSegNet_S  | 0.8524 | 0.9831 |

### Results on UCSD Background Subtraction dataset
Tables below show overall *test results* across 18 video sequences.

**For 20% split**

| Methods  | PWC(th=0.4) | F-Measure(th=0.4) | PWC(th=0.7) | F-Measure(th=0.7) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| FgSegNet_M  | 0.6260 | 0.8948 | 0.6381 | 0.8912 |
| FgSegNet_S  | 0.7052 | 0.8822 | 0.6273 | 0.8905 |

**For 50% split**

| Methods  | PWC(th=0.4) | F-Measure(th=0.4) | PWC(th=0.7) | F-Measure(th=0.7) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| FgSegNet_M  | 0.4637 | 0.9203 | 0.4878 | 0.9151 |
| FgSegNet_S  | 0.5024 | 0.9139 | 0.4676 | 0.9149 |

## Updates
**07/08/2018:**
- combine ```FgSegNet_S``` with ```FgSegNet_M``` and more
- add SBI2015, UCSD dataset and more results
- add published paper and a new citation
- integrate custom loss and other codes in a single file, for ease of use
- other improvements

**09/06/2018:**
- add quantity results on *changedetection.net* (CDnet2014 dataset)
- add supporting codes

**29/04/2018:**
- add Jupyter notebook and a model for testing (FgSegNet_M).

**27/01/2018:**
- add FgSegNet_M (a triplet network) source code and training frames

## Contact
lim.longang at gmail.com <br/>
Any issues/discussions are welcome.
