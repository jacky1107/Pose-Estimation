# poseEstimate

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
matplotlib
sklearn
```

Run the program
```
python3 main.py
```

# Notes

In this project, I divided it into 5 parts.

1. Extract the human pose features

The following link is open source(Named AlphaPose which is famous human body pose feature extraction tool) that help me to extract the pose features. (https://github.com/Amanbhandula/AlphaPose)

2. Design the reliable features



3. Build the classification(SVM)

4. Prediction

5. Evalution

(a) what other sources you used apart from the lecture material used in class during your work on the assignment

(b) how to compile and run your program

(c) any interesting features and extensions of your assignment.

# Reference
https://github.com/Amanbhandula/AlphaPose

http://human-pose.mpi-inf.mpg.de/contents/pishchulin14gcpr.pdf
