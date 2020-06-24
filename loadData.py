import os
import json
import numpy as np

def loadPoseData():
    folderName = "json"
    files = os.listdir(folderName)
    results = []

    for file in files:
        name = os.path.join(folderName, file)
        with open(name) as f:
            data = json.load(f)
            frames = len(data)
            outputs = np.zeros((frames, 17, 3))
            print(name, frames)
            for frame in range(frames):
                poseData = data[frame]["keypoints"]
                poseData = np.array(poseData)
                poseData = poseData.reshape((17, 3))
                outputs[frame] = poseData
            results.append(outputs)
    
    return np.array(results)
