# -*- coding: utf-8 -*-
from pyramid import pyramid, silding_window
import os
from non_max_suppression import nms
from detector import BOW
import numpy as np
import cv2
import csv

"""
Created on Thu Oct 18 15:16:20 2018

@author: zy
"""

'''
使用BOW+SVM进行滑动窗口目标检测
可以分类
'''


def prepare_data(rootpath0):
    '''
    加载数据集  
    args：
        rootpath：数据集所在的根目录
                  要求在该路径下，存放数据，每一类使用一个文件夹存放，文件名即为类名
        sample：指定获取的每一类样本长度

    return：
        train_path：训练集路径 list类型  [['calss0-1','calss0-2','calss0-3','class0-4',...]
                                         ['calss1-1','calss1-2','calss1-3','class1-4',...]
                                         ['calss2-1','calss2-2','calss2-3','class2-4',...]
                                         ['calss3-1','calss3-2','calss3-3','class3-4',...]                                        
        labels：每一个样本类别标签 list类型 [[0,0,0,0]...
                                            [1,1,1,1]...
                                            [2,2,2,2]...
                                            [3,3,3,3]...                                            
                                            ...]
        classes：每一个类别对应的名字 list类型
    '''
    files = [[], [], [], [], [], [], [], [], [], []]
    labels = [[], [], [], [], [], [], [], [], [], []]
    # 获取rootpath下的所有文件夹
    classes = ['01_palm', '02_l', '03_fist',  '04_fist_moved',
               '05_thumb', '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']
    # 遍历每个类别样本

    for i in range(10):
        rootpath = rootpath0 + '0' + str(i)
        for idx in range(len(classes)):
            # 获取当前类别文件所在文件夹的全路径
            path = os.path.join(rootpath, classes[idx])
            # print(path)
            # 遍历每一个文件路径
            filelist = [os.path.join(path, x) for x in os.listdir(
                path) if os.path.isfile(os.path.join(path, x))]

            # 追加到字典
            files[idx].extend(filelist)
            labels[idx].extend([idx+1]*200)

    # 返回files为10×2000，10种手势，每种2000个数量
    # 返回labels对应于files
    
    print("数据集加载完毕：共10种手势，每种"+str(len(files[0]))+"个。")
    return files, labels, classes


if __name__ == '__main__':
    '''
    1、训练或者直接加载训练好的模型
    '''
    # 训练？
    is_training = True
    bow = BOW()
    files = []
    labels = []

    if is_training:
        # 用来训练的样本的个数，样本个数 越大，训练准确率相对越高
        samples = 5
        # 根路径
        rootpath = 'leapGestRecog/'
        # 训练集图片路径
        filesAll, labelsAll, classes = prepare_data(rootpath)

        for i in range(len(filesAll)):
            files.append(filesAll[i][:samples])
            labels.append(labelsAll[i][:samples])

        print("训练集图片，每类各有"+str(len(files[0]))+"个样本。")

        # k越大，训练准确率相对越高
        trainData, trainLabels = bow.fit(files, labels, 20, samples//5)
        print(trainLabels)
        # 保存模型
        bow.save('svm.mat')
        print("length of trainData:", len(trainData), len(trainData[0]))
        print("length of trainLabels:", len(trainLabels))

        
        with open('trainData.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in trainData:
                writer.writerow(row)

        with open('trainLabels.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(trainLabels)

    else:
        # 加载模型
        bow.load('svm.mat')

    '''
    2、测试计算目标识别的准确率
    '''
    # 测试样本数量，测试结果
    # start_index = 1800
    # test_samples = 200
    # test_results = []

    # #指定测试图像路径
    # #根路径
    # rootpath = 'leapGestRecog/'
    # #训练集图片路径

    # for j in range(len(filesAll)):
    #     for i in range(start_index,start_index+test_samples):
    #         #预测
    #         print("正在预测："+str(filesAll[j][i]))

    #         img = cv2.imread(filesAll[j][i])
    #         label,score = bow.predict(img)
    #         if label is None:
    #             continue
    #         #print(files[j][i],label,labels[j][i])
    #         if label == labels[j][i]:
    #             test_results.append(True)
    #         else:
    #             test_results.append(False)

    # test_results = np.asarray(test_results,dtype=np.float32)
    # #计算准确率
    # accuracy = np.mean(test_results)
    # print('测试准确率为：',accuracy)

    '''
    3、利用滑动窗口进行目标检测
    '''
    # 滑动窗口大小
    # w,h = 600,600
    # test_img = '1.jpg'

    # img = cv2.imread(test_img)
    # rectangles = []
    # counter = 1
    # scale_factor =  10
    # font = cv2.FONT_HERSHEY_PLAIN

    # #label,score = bow.predict(img[50:280,100:280])
    # #print('预测：',label,score)

    # #图像金字塔
    # for resized in pyramid(img.copy(),scale_factor,(img.shape[1]//20,img.shape[1]//20)):
    #     print(resized.shape)
    #     #图像缩小倍数
    #     scale = float(img.shape[1])/float(resized.shape[1])
    #     #遍历每一个滑动区域
    #     for (x,y,roi) in silding_window(resized,10,(w,h)):
    #         if roi.shape[1] != w or roi.shape[0] != h:
    #             continue
    #         try:
    #             label,score = bow.predict(roi)
    #             #识别为人
    #             if label == 0:
    #                 #得分越小，置信度越高
    #                 if score < -1:
    #                     #print(label,score)
    #                     #获取相应边界框的原始大小
    #                     rx,ry,rx2,ry2 = x*scale,y*scale,(x+w)*scale,(y+h)*scale
    #                     rectangles.append([rx,ry,rx2,ry2,-1.0*score])
    #         except:
    #            pass
    #         counter += 1

    # windows = np.array(rectangles)
    # boxes = nms(windows,0.15)

    # for x,y,x2,y2,score in boxes:
    #     cv2.rectangle(img,(int(x),int(y)),(int(x2),int(y2)),(0,0,255),1)
    #     cv2.putText(img,'%f'%score,(int(x),int(y)),font,1,(0,255,0))

    # print("finished...")
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
