import cv2
import json
import math
import numpy as np
import matplotlib.pyplot as plt

from config import *
from training import *
from loadData import *

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

frameStep = 15 # 15
points = [9, 10, 15, 16] # 9, 10, 15, 16
steps = [400, 100, 30, 30, 20, 80] # 400, 100, 30, 30, 20, 80
labels = ["run", "walk", "dance"]

print(f"#=====Loading data=====#")
files, train_data = training(steps, points)
x_train, y_train = trainTestSplit(train_data)

print(f"x_train: {x_train.shape}")
print(f"y_train: {y_train.shape}")

print(f"#=====Training=====#")
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(x_train, y_train)
print(f"Done")
print("")

cap = cv2.VideoCapture("dataSet/test_data.mp4")
filePath = "test_data.json"
allData = loadPoseData(filePath)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(3)),
        int(cap.get(4)))
videoWriter = cv2.VideoWriter('results.mp4', fourcc, fps, size)

print(f"#=====Testing=====#")
results = []
count = 0
x1, x2 = 0, 0
correct, wrong = 0, 0
for i in range(len(allData)):
    ret, frame = cap.read()

    if x2 < frameStep: x1 = 0
    else: x1 += 1

    if (x2 - x1) == (frameStep - 1):
        data = allData[x1:x2, :, :2]

        testData = []
        for point in points:
            pose = data[:, point, :2]
            std = stdDev(pose)
            testData.append(std)
        output = int(clf.predict([testData]))
        results.append(output)
        if len(results) > 2:
            if results[0] == results[-1]: results[1] = results[0]
            results = results[1:]

        outputClass = labels[results[0]]
        cv2.putText(frame, str(outputClass), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)
        
        #=========ground truth=========#
        if count < 104: truth = 1
        elif count < 214: truth = 0
        elif count < 265: truth = 1
        elif count < 363: truth = 2
        elif count < 463: truth = 1
        else: truth = 0

        #===========evaluate===========#
        if output == truth: correct += 1
        else: wrong += 1

    x2 += 1
    count += 1
    loading = round(count/len(allData), 2)
    print(f"\rtesting: {loading}", end="")
    videoWriter.write(frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break


print("\n")
print(f"#=====Evaluate=====#")
total = correct + wrong
acc = round(correct / total, 2)

print(f"Steps   Files")
for i in range(len(files)):
    print(f"{steps[i]}    {files[i]}")
print(f"Correct: {correct}")
print(f"Wrong: {wrong}")
print(f"Acc: {acc}")

cap.release
videoWriter.release
cv2.destroyAllWindows()