# AI-Project
AI summer project

*To download the project we recommend using [this](https://github.com/Snehee2901/AI-Project/archive/refs/heads/main.zip) link. This downloads the latest version of the repository.* 

# Contents

1. [Introduction](#Introduction) 
2. [Project Description](#Project-Description)
3. [Downloading Dataset](#Downloading-Dataset)
4. [How to run](#How-to-run)
5. [Acknowledgement](#Acknowledgement)

# Introduction
The present project was undertaken as a component of the Applied Artificial Intelligence (COMP6721) course that was taught at Concordia University during the Summer 2023 term. The main objective of this study is to conduct a comparative analysis of various established Artificial Intelligence methodologies such as Decision Tree Classifier, Semi-Supervised Decision Tree Classifier and Convolutional Neural Network (CNN) architectures in the context of Histopathological Image Classification for Lung and Colon Cancer.

# Project Description

This project focuses on the classification of lung and colon tissue images into different benign and malignant categories using three different approaches: 
<ul>
<li>Decision tree</li>
<li>Semi-supervised decision tree.</li>
<li>Convolutional neural network (CNN).</li>
</ul>

The project utilizes the Lung and Colon Cancer Histopathological Images dataset, which consists of 25,000 histopathological images of lung and colon tissues with five classes. The dataset is divided into training, validation, and test sets, and preprocessing steps such as resizing and normalization are applied to the images.

The decision tree algorithm constructs a classification model based on labeled data, while the semi-supervised decision tree incorporates both labeled and unlabeled data during the training phase. The CNN architecture is specifically designed for image classification and utilizes convolutional and fully connected layers to extract intricate patterns and features.

The project compares the performance of these three approaches in accurately classifying tissue images. The results show that the CNN approach achieves the highest accuracy, surpassing the traditional decision tree and semi-supervised decision tree models. This highlights the effectiveness of deep learning techniques, such as CNNs, in learning complex image representations and outperforming traditional machine learning methods in tissue image classification tasks.

# Downloading Dataset

An overview of the dataset used in this study is available [here](https://arxiv.org/abs/1912.12142v1). This dataset is also available to download from the kaggle and can be downloaded through this [link](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images). The sie of the dataset is approximately 2GB and it contains 25000 Images which are spread equally among 5 classes of the Lung adnd Colon cancer. After downloading, the dataset can be extracted to folder where the source files are being kept. Next, the code to split the dataset into train, test, and validation subsets are already present in all 3 source files.

# How to run

## Installing Requirements
You can find all the requirements to run the project in [this](https://github.com/Snehee2901/AI-Project/blob/main/requirements.txt). To install the requirement run the following command:

```
pip install -r requirements.txt
```

## How to train/validate model
The jupyter notebooks with their saved respective models can be found from below.

| Methodolgy                    | Link to notebook                                                                                       | Link to model                                                                              |
|-------------------------------|--------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Supervised Decision Tree      | [notebook](https://github.com/Snehee2901/AI-Project/blob/main/Supervised%20Decision%20tree.ipynb)      | [model](https://github.com/Snehee2901/AI-Project/blob/main/DecisionTree.pkl)               |
| Semi-supervised Decision Tree | [notebook](https://github.com/Snehee2901/AI-Project/blob/main/Semi-supervised%20Decision%20Tree.ipynb) | [model](https://github.com/Snehee2901/AI-Project/blob/main/SemiSupervisedDecisionTree.pkl) |
| Convolutional Neural Network  | [notebook](https://github.com/Snehee2901/AI-Project/blob/main/CNN.ipynb)                               | [model](https://github.com/Snehee2901/AI-Project/blob/main/CNN.pt)                         |

### Train from scratch
To train the model from scratch you can download the notebook from the respective methodology and save it in the same folder as that of the dataset and you can execute the notebook, it will start the training, validating and testing the model.

### Validate the presaved Model
To test the presaved model you can follow the below given steps
1. To load the model you will require the library *pickle*, you can download it using the following code 
```
pip install pickle
```
2. Use the pickle to load the model in the new file
```
import pickle

# Specify the path to the saved model
model_path = "path_to/model.pkl"

# Load the saved model
with open(model_path, "rb") as file:
    model = pickle.load(file)
```

3. Using the LC25000 dataset you can generate your own validation dataset

4. Using the methods of the scikit learn or pytorch (depending on the Decision tree or Neural Network) you can test the dataset and interpret the metrics.


# Achnowledgement
We would like to thank Dr. Arash Azarfar for the highly effective course instruction, as well as to all of the Teaching Assistants who provided guidance and support throughout the project.
