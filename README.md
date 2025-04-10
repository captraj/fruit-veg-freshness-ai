# A Simple Freshness Classification Model (Using MobileNet V2)

This repository contains code for the creation, training and implementation of a model that classifies the freshness of a fruit or vegetable image as either fresh, medium fresh, or not fresh and also provides a freshness index based on the assessed image. The model is implemented in Python using Tensorflow and OpenCV libraries and uses a Transfer Learning approach by using MobileNet V2 pretrained model.

## Model Description

The model has been trained using a Jupyter Notebook (freshness_regression.ipynb) to classify the freshness of input fruit or vegetable images. The training process involves the use of a preprocessed dataset and a custom convolutional neural network (CNN) architecture based on the MobileNetV2 model.

The MobileNetV2 model is a powerful feature extractor that has been pre-trained on the ImageNet dataset. We use this model as the base and freeze its layers to prevent them from being modified during our training process. On top of the MobileNetV2 model, we add additional layers to fine-tune the model for our specific freshness classification task.

## Model Architecture

The model architecture is as follows:

1. MobileNetV2 (up to the last convolutional layer) as the base feature extractor.
2. BatchNormalization layer to normalize the outputs from the base model.
3. Two SeparableConv2D layers with 64 filters each, followed by ReLU activation for feature extraction.
4. MaxPooling2D layer with a pool size of (2, 2) to downsample the spatial dimensions.
5. A Dropout layer with a dropout rate of 40% to reduce overfitting.

6. Two Conv2D layers with 128 filters each, followed by ReLU activation for further feature extraction.
7. BatchNormalization layer to normalize the outputs from the previous layers.
8. MaxPooling2D layer with a pool size of (2, 2) and Dropout layer with a dropout rate of 50%.

9. Flatten layer to convert the 2D feature maps into a 1D feature vector.
10. Dense layer with 128 units and ReLU activation for high-level feature learning.
11. Dropout layer with a dropout rate of 30% to further prevent overfitting.

12. Dense output layer with 1 unit and a sigmoid activation function for binary classification (fresh or not fresh).

## Requirements

Before using the model, ensure you have the following dependencies installed:

- Python 3.10
- Tensorflow
- OpenCV
- Numpy

You can install the required libraries by using the following commands for conda environment:

```bash
conda install -c conda-forge tqdm -y
conda install -c conda-forge matplotlib -y
conda install -c conda-forge pandas -y
conda install -c conda-forge opencv -y
```

## Usage of evaluation script

To use the model for classifying the freshness of a fruit or vegetable image, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/captraj/fruit-veg-freshness-ai.git
cd fruit-veg-freshness-ai
```


2. Make sure to place the image you want to evaluate in the repository's root directory and rename it to `image-to-eval.png`. You may replace `image-to-eval.png` with the path to your own image if it's located elsewhere.

3. Use the `evaluate-image.py` script to evaluate the freshness of the image:

```bash
python evaluate-image.py
```


4. The script will output the prediction and its freshness classification:

`Prediction: 0.245`
`The item is MEDIUM FRESH`

Here, the value `0.245` represents the model's confidence that the item is fresh. The classification is determined based on predefined thresholds.

### Customization

If you wish to customize the thresholds used for freshness classification, you can do so by modifying the values of `threshold_fresh` and `threshold_medium` in the `evaluate-image.py` script. Adjusting these values according to your standards may lead to better predictions for your specific use case.

This project has been completed!

The further implementation of this project into an API is listed on my profile as **freshcheck**, do have a look at that to better understand the integration of this repository.

The **FreshCheck** repository is located at : [https://github.com/captraj/freshcheck]









