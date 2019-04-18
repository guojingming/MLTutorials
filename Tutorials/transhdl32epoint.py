import cv2
import math
import numpy as np
import struct

kittiDataPath = "/home/jlurobot/HDL32E"

pointPath = kittiDataPath + "/"

def pointProcess(pointFile):
    str = pointFile.read()
    lenStr = len(str)
    points = struct.unpack("%df"%(lenStr/4), str)
    pointsArray = np.array(points)
    pointsArray = pointsArray.reshape((-1, 4))
    return pointsArray

r_total_min = 1000000
r_total_max = -1000000
c_total_min = 1000000
c_total_max = -1000000
delta_theta_32 = 0.2
delta_fi_32 = 1.33
pi = 3.1415926
scale_factor = delta_fi_32 / delta_theta_32
c_scale_factor = delta_theta_32 / delta_fi_32 * scale_factor
r_scale_factor = scale_factor

for index in range(1000):
    pointFile = open(pointPath + "{number:06}.bin".format(number=index), "rb")
    pointsArray = pointProcess(pointFile)
    r_min = 1000000
    r_max = -1000000
    c_min = 1000000
    c_max = -1000000
    img = np.zeros((600 * 1200 * 3)).reshape(600, 1200, 3)

    for point in pointsArray:
        x = point[0]
        y = point[1]
        z = point[2]
        i = point[3]

        if x < 0:
            continue

        c = math.atan2(y, x) / (delta_theta_32 * pi / 180)
        r = math.atan2(z, math.sqrt(x * x + y * y)) / (delta_fi_32 * pi / 180)
        color = 255 * i / 20

        if r_max < r:
            r_max = r
        if r_min > r:
            r_min = r
        if c_max < c:
            c_max = c
        if c_min > c:
            c_min = c

        index_x = int(-1 * c_scale_factor * c + 600)
        index_y = int(-1 * r_scale_factor * r + 300)
        if index_y >= 0 and index_y < 600 and index_x >= 0 and index_x < 1200:
            img[index_y][index_x][0] = (255 - color) / 255
            img[index_y][index_x][1] = 100 / 255
            img[index_y][index_x][2] = color / 255

    print("{0} {1} {2} {3}".format(r_max, r_min, c_max, c_min))
    cv2.imshow("IMG", img)
    cv2.waitKey(0)
