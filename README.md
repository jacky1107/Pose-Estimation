# PoseEstimate

author: Jacky Wang

All code is written in Python, if there is any question please feel free to ask.

# Introdution
In this project I consider the task of human activity recognition in realistic videos. Therefore, I made the videos that including dancing, walking and running in "dataSet" folder. Each activity has 2 videos and total is 6. Furthermore, in order to evaluate performance I made the test video that contains 3 different activities, also, each frame has been labeled.

# Implementation

Please make sure you have installed these packages.

```
cv2
json
math
numpy
sklearn
matplotlib
```

Run the program
```
python3 main.py
```

# Notes

In this project, I divided it into 5 parts.

1. Extract the human pose features

The following link is an open-source(Named AlphaPose which is a famous tool to extract feature from human body) that helps me to extract the human body features. (https://github.com/Amanbhandula/AlphaPose)

2. Design the reliable features

We know that there are a lot of diffrences among the action of walking, running and dancing. For example, when we are walking, our hands' motion must be smaller than running. First, we calculated standard deviation with the points we collected as relible features we can use. That means, we can get a high standard deviation from a large movement. Then, we need to calculate distance among each point to find the boundary.

3. Build the classification(SVM)

I choose the SVM to be the classification. Support Vector Machine can find the boundary and maximize the margin of each class. Thus, SVM is a proper classification in this case.

4. Evalution

In conclusion, the accuracy is 86%, which is a acceptable that we expected. However, we still had lots of doubts to this case, such as "Is SVM a stable classification for this case?". To answer that, we implemented a KNN algorithm as a new classification. Compared to SVM, KNN was worse, which only had 70% accuracy. There must be other classification are better than SVM, We are trying to find out the optimal method.

# Reference
https://github.com/Amanbhandula/AlphaPose

http://human-pose.mpi-inf.mpg.de/contents/pishchulin14gcpr.pdf
