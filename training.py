import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import *
from loadData import *

def trainTestSplit(train_data):
    size = len(train_data)
    x_train = []
    y_train = []
    for features, label in train_data:
        x_train.append(features)
        y_train.append(label)
    return np.array(x_train), np.array(y_train)

def training(steps, points):
    train_data = []
    folderName = "poseDataJson"
    files = os.listdir(folderName)

    for index, file in enumerate(files):
        step = steps[index]
        if "run" in file: label = 0
        elif "walk" in file: label = 1
        elif "dance" in file: label = 2

        path = os.path.join(folderName, file)
        allData = loadPoseData(path)

        frames, featureDimension, info = allData.shape
        x1 = 0
        for x2 in range(step, frames, 1):
            feature = []

            data = allData[x1:x2, :, :2]
            for point in points:
                pose = data[:, point, :2]
                std = stdDev(pose)
                feature.append(std)
            train_data.append([feature, label])
            x1 += 1

    return files, train_data