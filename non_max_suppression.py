# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:37:42 2018

@author: zy
"""
import numpy as np

'''
非极大值抑制
https://blog.csdn.net/hongxingabc/article/details/78996407

1、按打分最高到最低将BBox排序 ，例如：A B C D E F

2、A的分数最高，保留，从B-E与A分别求重叠率IoU，假设B、D与A的IoU大于阈值，
   那么B和D可以认为是重复标记去除

3、余下C E F，重复前面两步

'''
def nms(boxes,threshold):
    '''
    对边界框进行非极大值抑制
    args:
        boxes：边界框，数据为list类型，形状为[n,5]  5位表示(x1,y1,x2,y2,score)
        threshold：IOU阈值  大于该阈值，进行抑制
    '''
    if len(boxes) == 0:
        return []
        
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    scores = boxes[:,4]
    
    #计算边界框区域大小，并按照score进行倒叙排序
    areas = (x2-x1 + 1)*(y2-y1 + 1)    
    idxs = np.argsort(scores)[::-1]

    #keep为最后保留的边框  
    keep = []  
    
    while len(idxs) > 0:  
        
        #idxs[0]是当前分数最大的窗口，肯定保留  
        i = idxs[0]  
        keep.append(i)
        
        #计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[idxs[1:]])  
        yy1 = np.maximum(y1[i], y1[idxs[1:]])  
        xx2 = np.minimum(x2[i], x2[idxs[1:]])  
        yy2 = np.minimum(y2[i], y2[idxs[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        
        #交/并得到iou值  
        ovr = inter / (areas[i] + areas[idxs[1:]] - inter)  
        
        #inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收  
        inds = np.where(ovr <= threshold)[0]  
        
        #order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        idxs = idxs[inds + 1]  

    return boxes[keep]