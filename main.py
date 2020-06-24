import os
import cv2
import json
import numpy as np

folderName = "json"
files = os.listdir(folderName)

for file in files[:1]:
    name = os.path.join(folderName, file)
    with open(name) as f:
        print(name)
        data = json.load(f)

        frames = len(data)
        for frame in range(frames):
            poseData = data[frame]["keypoints"]
            print(poseData)

        # print(poseData)
        # print(len(poseData) / 3)
