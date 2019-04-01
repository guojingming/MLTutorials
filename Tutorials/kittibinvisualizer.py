import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import struct
import os
import subprocess

def pointProcess(pointFile):
    str = pointFile.read()
    lenStr = len(str)
    points = struct.unpack("%df"%(lenStr/4), str)
    pointsArray = np.array(points)
    pointsArray = pointsArray.reshape((-1, 4))
    return pointsArray


kittiDataPath = "/home/jlurobot/Kitti/object/training"

pointPath = kittiDataPath + "/velodyne/"

pointLenth = len([name for name in os.listdir(pointPath) if os.path.isfile(os.path.join(pointPath, name))])


for index in range(pointLenth):
    pointFile = open(pointPath + "{number:06}.bin".format(number=index), "rb")
    pointProcessResult = pointProcess(pointFile)
    points = pointProcessResult
    skip = 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    point_range = range(0, points.shape[0], skip)

    ax.scatter(points[point_range, 0],
               points[point_range, 1],
               points[point_range, 2],
               c=points[point_range, 2],
               cmap='Spectral',
               marker=".")
    ax.axis('scaled')
    plt.show()

    pointFile.close()
    input()

