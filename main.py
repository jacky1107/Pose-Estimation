import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import *
from loadData import *

poseData = loadPoseData()
points = [9, 10, 15, 16]

mp4 = ["walk", "run", "dance"]
for mp4Index in range(3):
    data = poseData[mp4Index]

    # for i in range(len(data)):
    #     center = data[i, 0, 1]
    #     print(center)

    for point in points:
        pose = data[:, point, :2]
        print(f"Mp4: {mp4[mp4Index]}, Points: {point}, Std Dev: {stdDev(pose)}")
        # plt.scatter(pose[:, 0], pose[:, 1])
    # plt.show()

newData = poseData[0][0]
cap = cv2.VideoCapture("videos/walk.mp4")
ret, frame = cap.read()

for i in range(17):
    cv2.putText(frame, str(i), (int(newData[i, 0]), int(
        newData[i, 1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("frame", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
