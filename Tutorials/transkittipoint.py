import cv2
import os
import math
import numpy as np
import struct
import config as cfg

#/home/jlurobot/Kitti
#   +-- object
#       +-- testing
#           +-- calib
#               +-- 000000.txt
#           +-- image_2
#               +-- 000000.png
#           +-- label_2
#               +-- 000000.txt
#           +-- velodyne
#               +-- 000000.bin
#       +-- training
#           +-- calib
#               +-- 000000.txt
#           +-- image_2
#               +-- 000000.png
#           +-- label_2
#               +-- 000000.txt
#           +-- velodyne
#               +-- 000000.bin
#           +-- planes
#               +-- 000000.txt
#       +-- train.txt
#       +-- trainval.txt
#       +-- val.txt
def calibProcess(calibFile):
    calibInfo = {}
    for line in calibFile.readlines():
        line = line.split(" ")
        if line[0] == "P2:":
            calibInfo["P2_array"] = np.array([[float(line[1]), float(line[2]), float(line[3]), float(line[4])],
                                                [float(line[5]), float(line[6]), float(line[7]), float(line[8])],
                                                [float(line[9]), float(line[10]), float(line[11]), float(line[12])]
                                            ])
            #calibInfo["P2_array"] = np.delete(calibInfo["P2_array"], 3, axis=1)
        elif line[0] == "Tr_velo_to_cam:":
            calibInfo["Tr_array"] = np.array([[float(line[1]), float(line[2]), float(line[3]), float(line[4])],
                                              [float(line[5]), float(line[6]), float(line[7]), float(line[8])],
                                              [float(line[9]), float(line[10]), float(line[11]), float(line[12])]
                                              ])
        elif line[0] == "R0_rect:":
            calibInfo["Rect0_array"] = np.array([[float(line[1]), float(line[2]), float(line[3])],
                                              [float(line[4]), float(line[5]), float(line[6])],
                                              [float(line[7]), float(line[8]), float(line[9])]
                                              ])
    return calibInfo

def imageProcess(imageFile):
    return [imageFile, ]

def labelProcess(labelFile):
    labelProcessResult = {}
    detectionLabels2d = []
    detectionLabels3d = []
    for line in labelFile.readlines():
        line = line.split(" ")
        tag_class = line[0]
        if tag_class == "DontCare":
            continue
        detectionLabel2d = []
        detectionLabel2d.append(tag_class)
        detectionLabel2d.append(int(float(line[4])))
        detectionLabel2d.append(int(float(line[5])))
        detectionLabel2d.append(int(float(line[6])))
        detectionLabel2d.append(int(float(line[7])))
        detectionLabels2d.append(detectionLabel2d)
        detectionLabel3d = []
        detectionLabel3d.append(tag_class)
        detectionLabel3d.append(float(line[8]))
        detectionLabel3d.append(float(line[9]))
        detectionLabel3d.append(float(line[10]))
        detectionLabel3d.append(float(line[11]))
        detectionLabel3d.append(float(line[12]))
        detectionLabel3d.append(float(line[13]))
        detectionLabel3d.append(float(line[14]))
        detectionLabels3d.append(detectionLabel3d)
    labelProcessResult["label2d"] = detectionLabels2d
    labelProcessResult["label3d"] = detectionLabels3d
    return labelProcessResult

def pointProcess(pointFile):
    str = pointFile.read()
    lenStr = len(str)
    points = struct.unpack("%df"%(lenStr/4), str)
    pointsArray = np.array(points)
    pointsArray = pointsArray.reshape((-1, 4))
    return pointsArray

def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
        angle = np.pi / 2
    return angle

def point3dTo2D(input3dPoints, calibInfo):
    # input N, 3
    T1 = calibInfo["Tr_array"]
    T1 = np.concatenate((T1, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    T2 = np.concatenate((calibInfo["Rect0_array"], np.array([0, 0, 0]).reshape(1, 3)), axis=0)
    T2 = np.concatenate((T2, np.array([0, 0, 0, 1]).reshape(4, 1)), axis=1)
    P2 = calibInfo["P2_array"]

    N = input3dPoints.shape[0]
    print("N: %d"%N)
    ins = input3dPoints[:, 3].reshape(N, 1)
    input3dPoints = np.concatenate((input3dPoints[:, 0:3], np.ones(N).reshape(N, 1)), axis=1)

    xMax = math.fabs(input3dPoints.max(axis=0)[0])
    xMin = math.fabs(input3dPoints.min(axis=0)[0])
    if xMax < xMin:
        xMax = xMin

    yMax = math.fabs(input3dPoints.max(axis=0)[1])
    yMin = math.fabs(input3dPoints.min(axis=0)[1])
    if yMax < yMin:
        yMax = yMin

    zMax = math.fabs(input3dPoints.max(axis=0)[2])
    zMin = math.fabs(input3dPoints.min(axis=0)[2])
    if zMax < zMin:
        zMax = zMin



    R = np.dot(np.dot(np.dot(P2, T2), T1), input3dPoints.T).T
    R[:, [0]] /= R[:, [2]]
    R[:, [1]] /= R[:, [2]]
    R = np.delete(R, 2, axis=1)

    R = np.concatenate((R, input3dPoints[:, 0:3]), axis=1)
    R = np.concatenate((R, ins), axis=1)
    # output (N, 6) and (1, 3)
    # (x_2d y_2d x_3d y_3d z_3d ins) (x_max y_max z_max)
    maxValues = np.array([xMax, yMax, zMax])
    resultMap = {}
    resultMap["R"] = R
    resultMap["max"] = maxValues
    return resultMap

def boxes3dTo2D(input3dBoxes, calibInfo):
    # input N, 7
    MeanVel2Cam = np.array(([
                            [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
                            [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
                            [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
                            [0, 0, 0, 1]
                        ]))
    MeanRect0 = np.array(([
                            [0.99992475, 0.00975976, -0.00734152, 0],
                            [-0.0097913, 0.99994262, -0.00430371, 0],
                            [0.00729911, 0.0043753, 0.99996319, 0],
                            [0, 0, 0, 1]
                        ]))
    T1 = calibInfo["Tr_array"]
    T1 = np.concatenate((T1, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    T2 = np.concatenate((calibInfo["Rect0_array"], np.array([0, 0, 0]).reshape(1, 3)), axis=0)
    T2 = np.concatenate((T2, np.array([0, 0, 0, 1]).reshape(4, 1)), axis=1)
    P2 = calibInfo["P2_array"]

    #N, 3
    N = input3dBoxes.shape[0]
    centerPoints = input3dBoxes[:, :3]
    #N, 4
    temp = np.ones((N, 1))
    centerPoints = np.concatenate((centerPoints, temp), axis=1)

    #(4, 4) * (4, N) = (4, N)
    res = np.matmul(np.linalg.inv(MeanRect0), centerPoints.T)
    #(4, 4) * (4, N) = (4, N)
    res = np.matmul(np.linalg.inv(MeanVel2Cam), res)
    #(4, N) -> (N, 4)
    res = res.T
    #(N, 4) -> (N, 3)
    res = res[:, 0:3]
    res = res.reshape(-1, 3)
    #(N, 4) h,w,l,r
    temp = input3dBoxes[:, 3:]
    #(N, 3) -> (N, 7) x,y,z,h,w,l,r in velodyne-coodinate
    res = np.concatenate((res, temp), axis=1)

    boxes2D = []
    for box in res:
        R1 = box[0:3].reshape(1, 3)
        R1 = R1.T
        h = box[3]
        w = box[4]
        l = box[5]
        r = angle_in_limit(-1 * box[6] - math.pi / 2)
        T3 = np.array([
            [-0.5 * l, -0.5 * l, 0.5 * l, 0.5 * l, -0.5 * l, -0.5 * l, 0.5 * l, 0.5 * l],
            [0.5 * w, -0.5 * w, -0.5 * w, 0.5 * w, 0.5 * w, -0.5 * w, -0.5 * w, 0.5 * w],
            [0, 0, 0, 0, h, h, h, h]
        ])
        T4 = np.array([
            [math.cos(r), -1 * math.sin(r), 0],
            [math.sin(r), math.cos(r), 0],
            [0, 0, 1]
        ])
        T5 = np.tile(R1, 8)
        T6 = (np.dot(T4, T3) + T5).T    #8 corner-point's coordinates
        T7 = np.concatenate((T6, np.ones((8, 1))), axis=1).T
        T8 = np.dot(np.dot(T2, T1), T7)
        T9 = np.dot(P2, T8).T
        T9[:, 0] /= T9[:, 2]
        T9[:, 1] /= T9[:, 2]
        T9 = T9[:, 0:2]
        boxes2D.append(T9)
    print(boxes2D)
    return boxes2D

def visualization(calibProcessResult, imageProcessResult, labelProcessResult, pointProcessResult):
    res = []
    detection2dImg = imageProcessResult[0]
    detection3dImg = imageProcessResult[0].copy()
    detection3dPointImg = imageProcessResult[0].copy()
    detection3dPointImg[:, :, :] = 0

    label2d = labelProcessResult["label2d"]
    label3d = labelProcessResult["label3d"]
    for label in label2d:
        cv2.rectangle(detection2dImg, (label[1], label[2]), (label[3], label[4]), (255, 0, 255), 2)
        cv2.putText(detection2dImg, label[0], (label[1]-5, label[2]-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

    labels3d = []
    for label in label3d:
        h = label[1]
        w = label[2]
        l = label[3]
        x = label[4]
        y = label[5]
        z = label[6]
        r = label[7]
        labels3d.append(np.array([x, y, z, h, w, l, r]))
    #N, 7  ->  N, 8, 2
    boxes2d = boxes3dTo2D(np.array(labels3d).reshape(-1, 7), calibProcessResult)
    for box in boxes2d:
        for i in range(4):
            cv2.line(detection3dImg, (int(box[i][0]), int(box[i][1])), (int(box[(i + 1) % 4][0]), int(box[(i + 1) % 4][1])), (255, 0, 255), 2)
            cv2.line(detection3dImg, (int(box[i][0]), int(box[i][1])), (int(box[i + 4][0]), int(box[i + 4][1])), (255, 0, 255), 2)
            cv2.line(detection3dImg, (int(box[i + 4][0]), int(box[i + 4][1])), (int(box[i % 3 + 5][0]), int(box[i % 3 + 5][1])), (255, 0, 255), 2)

            cv2.line(detection3dPointImg, (int(box[i][0]), int(box[i][1])),(int(box[(i + 1) % 4][0]), int(box[(i + 1) % 4][1])), (255, 0, 255), 2)
            cv2.line(detection3dPointImg, (int(box[i][0]), int(box[i][1])), (int(box[i + 4][0]), int(box[i + 4][1])),(255, 0, 255), 2)
            cv2.line(detection3dPointImg, (int(box[i + 4][0]), int(box[i + 4][1])),(int(box[i % 3 + 5][0]), int(box[i % 3 + 5][1])), (255, 0, 255), 2)

    resultMap = point3dTo2D(pointProcessResult, calibProcessResult)
    points2d = resultMap["R"]
    maxValue = resultMap["max"]
    for point2d in points2d:
        w = detection3dPointImg.shape[1]
        h = detection3dPointImg.shape[0]
        if int(point2d[1]) >= 0 and int(point2d[1]) < h and int(point2d[0]) >=0 and int(point2d[0]) < w:
            #b,g,r
            detection3dPointImg[int(point2d[1])][int(point2d[0])][0] = 255 * (1 - point2d[5])
            detection3dPointImg[int(point2d[1])][int(point2d[0])][1] = 128 * (1 - math.fabs(point2d[2]) / maxValue[0])
            detection3dPointImg[int(point2d[1])][int(point2d[0])][2] = 255 * 1.2 * point2d[5]
    res.append(detection2dImg)
    res.append(detection3dImg)
    res.append(detection3dPointImg)
    return res

kittiDataPath = "/home/jlurobot/Kitti/object/training"

calibPath = kittiDataPath + "/calib/"
imagePath = kittiDataPath + "/image_2/"
labelPath = kittiDataPath + "/label_2/"
pointPath = kittiDataPath + "/velodyne/"

calibLenth = len([name for name in os.listdir(calibPath) if os.path.isfile(os.path.join(calibPath, name))])
imageLenth = len([name for name in os.listdir(imagePath) if os.path.isfile(os.path.join(imagePath, name))])
labelLenth = len([name for name in os.listdir(labelPath) if os.path.isfile(os.path.join(labelPath, name))])
pointLenth = len([name for name in os.listdir(pointPath) if os.path.isfile(os.path.join(pointPath, name))])

if not (calibLenth == imageLenth == labelLenth == pointLenth):
    print("Broken kitti data: The amounts of files in multi-folders are not unified.")
    print("Calib files count: {0}".format(calibLenth))
    print("Image files count: {0}".format(imageLenth))
    print("Label files count: {0}".format(labelLenth))
    print("Point files count: {0}".format(pointLenth))
    os._exit(1)
else:
    print("The amount of files is {0}".format(calibLenth))


for index in range(calibLenth):
    calibFile = open(calibPath + "{number:06}.txt".format(number=index), "r")
    imageFile = cv2.imread(imagePath + "{number:06}.png".format(number=index))
    labelFile = open(labelPath + "{number:06}.txt".format(number=index), "r")
    pointFile = open(pointPath + "{number:06}.bin".format(number=index), "rb")

    calibProcessResult = calibProcess(calibFile)
    imageProcessResult = imageProcess(imageFile)
    labelProcessResult = labelProcess(labelFile)
    pointProcessResult = pointProcess(pointFile)
    calibFile.close()
    labelFile.close()
    pointFile.close()

    visualizationResult = visualization(calibProcessResult, imageProcessResult, labelProcessResult, pointProcessResult)

    cv2.imshow("2D detection in img", visualizationResult[0])
    cv2.imshow("3D detection in img", visualizationResult[1])
    cv2.imshow("3D detection in points fv", visualizationResult[2])
    #cv2.imshow("2D detection in points bev", visualizationResult[3])


    calibFile.close()
    labelFile.close()
    pointFile.close()

    cv2.waitKey(0)