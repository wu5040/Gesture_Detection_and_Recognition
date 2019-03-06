from test import prepare_GestData

# samples = 100
# # 根路径
# rootpath = './Gesture_Detection_and_Recognition/GestData/'
# # # 训练集图片路径
# files, labels = prepare_GestData(rootpath)


# print(len(files[0]),len(labels))
# print(files[0][1])
# print(labels[9][2])
# print(len(files[0]),len(labels[0]))
# print(files[9][242])
# print(labels[9][242])
# print(classes)


# import csv

# List=[[1,2,3],[2,3,4],[4,3,5]]
# with open('List.csv', 'w', newline='') as csvfile:
#     writer  = csv.writer(csvfile)
#     for row in List:
#         writer.writerow(row)

import cv2
import numpy as np

img=cv2.imread("GestData/1/gjq_1_336.jpg",cv2.IMREAD_COLOR)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img1=cv2.imread("GestData/1/gjq_1_336.jpg",cv2.IMREAD_COLOR)
gray1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

detector=cv2.xfeatures2d.SURF_create()
ee=cv2.xfeatures2d.SURF_create()

detector1=cv2.xfeatures2d.SURF_create()
ee1=cv2.xfeatures2d.SURF_create()

keypoints=detector.detect(gray,None)
cv2.drawKeypoints(gray,keypoints,img)

keypoints1=detector1.detect(gray,None)
cv2.drawKeypoints(gray1,keypoints1,img1)
cv2.imshow('test',img1)
cv2.waitKey(0)
# print(keypoints)
print(len(ee1.compute(img1,keypoints1)[1]))
cv2.imshow('test',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import matplotlib.pyplot as plt
# x = [20, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 1000, 1500]
# y = [40.85831863609641,
#      50.91122868900647,
#      58.671369782480895,
#      60.08230452674898,
#      61.140505584950034,
#      62.08112874779541,
#      64.55026455026454,
#      65.02057613168725,
#      65.78483245149911,
#      66.01998824221046,
#      66.78424456202234,
#      68.72427983539094,
#      69.4]


# plt.plot(x,y,marker="o")
# plt.xlabel("k")
# plt.ylabel("accuracy")
# plt.show()
