import os
import json
import numpy as np

def loadPoseData(path):
    with open(path) as f:
        data = json.load(f)
        frames = len(data)
        outputs = np.zeros((frames, 17, 3))
        for frame in range(frames):
            poseData = data[frame]["keypoints"]
            poseData = np.array(poseData)
            poseData = poseData.reshape((17, 3))
            outputs[frame] = poseData
    return outputs
