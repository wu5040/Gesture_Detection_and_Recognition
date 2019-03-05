# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 14:59:09 2018

@author: zy
"""
'''
词袋模型BOW+SVM 目标检测

'''
import numpy as np
import cv2
import pickle
import os

class BOW(object):
    
    def __init__(self,):
        #创建一个SIFT对象  用于关键点提取
        self.feature_detector  = cv2.xfeatures2d.SIFT_create()
        #创建一个SIFT对象  用于关键点描述符提取 
        self.descriptor_extractor = cv2.xfeatures2d.SIFT_create()

    def fit(self,files,labels,k,length=None):
        '''
        开始训练 可以用于多分类
        
        args：
            files：训练集图片路径 
            labes：对应的每个样本的标签
            k：k-means参数k 
            length：指定用于训练词汇字典的样本长度 length<=samples
        '''
        #类别数
        classes = len(files)
        
        #样本数量
        samples = len(files[0])  
        
        if length is None:
            length = samples        
        elif  length > samples:
            length = samples
        
        #FLANN匹配  参数algorithm用来指定匹配所使用的算法，可以选择的有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneIndex，这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
        flann_params = dict(algorithm=1,tree=5)
        flann = cv2.FlannBasedMatcher(flann_params,{})
        
        #创建BOW训练器，指定k-means参数k   把处理好的特征数据全部合并，利用聚类把特征词分为若干类，此若干类的数目由自己设定，每一类相当于一个视觉词汇
        bow_kmeans_trainer = cv2.BOWKMeansTrainer(k)
                
        print('building BOWKMeansTrainer...')
        #合并特征数据  每个类从数据集中读取length张图片，通过聚类创建视觉词汇         
        for j in range(classes):        
            for i in range(length):                  
                #有一些图像会抛异常,主要是因为该图片没有sift描述符
                print("building BOWKMeansTrainer: ",j+1,i+1,"/",length)                
                descriptor = self.sift_descriptor_extractor(files[j][i])                                                          
                if not descriptor is None:
                    bow_kmeans_trainer.add(descriptor)                
                    #print('error:',files[j][i])
                    

        #进行k-means聚类，返回词汇字典 也就是聚类中心
        self.voc = bow_kmeans_trainer.cluster()
        
        #输出词汇字典  <class 'numpy.ndarray'> (40, 128)
        print("输出词汇字典:",type(self.voc),self.voc.shape)
        
        #初始化bow提取器(设置词汇字典),用于提取每一张图像的BOW特征描述
        self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.descriptor_extractor,flann)        
        self.bow_img_descriptor_extractor.setVocabulary(self.voc)        
        
        print('adding features to svm trainer...')
        
        #创建两个数组，分别对应训练数据和标签，并用BOWImgDescriptorExtractor产生的描述符填充
        #按照下面的方法生成相应的正负样本图片的标签
        traindata,trainlabels = [],[]
        for j in range(classes):
            for i in range(samples):
                print("adding features to svm trainer: ", j+1,i+1,"/samples")                   
                descriptor = self.bow_descriptor_extractor(files[j][i])
                if not descriptor is None:
                    traindata.extend(descriptor)
                    trainlabels.append(labels[j][i])                
                            
         
        # #创建一个SVM对象    
        # self.svm = cv2.ml.SVM_create()
        # self.svm.setType(cv2.ml.SVM_C_SVC)
        # self.svm.setGamma(0.5)
        # self.svm.setC(30)
        # self.svm.setKernel(cv2.ml.SVM_RBF)
        # #使用训练数据和标签进行训练
        # self.svm.train(np.array(traindata),cv2.ml.ROW_SAMPLE,np.array(trainlabels))

        return  traindata,trainlabels
        
        
    def save(self,path):
        '''
        保存模型到指定路径
        '''
        print('saving  model....')
        #保存svm模型
        # self.svm.save(path)
        #保存bow模型
        f1 = os.path.join(os.path.dirname(path),'dict.pkl')
        with open(f1,'wb') as f:
            pickle.dump(self.voc,f)

        
    def load(self,path):
        '''
        加载模型
        '''
        print('loading  model....')
        #加载svm模型
        self.svm = cv2.ml.SVM_load(path)
        
        #加载bow模型                
        f1 = os.path.join(os.path.dirname(path),'dict.pkl')
        with open(f1,'rb') as f:
            voc = pickle.load(f)
            #FLANN匹配  参数algorithm用来指定匹配所使用的算法，可以选择的有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneIndex，这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
            flann_params = dict(algorithm=1,tree=5)
            flann = cv2.FlannBasedMatcher(flann_params,{})
            #初始化bow提取器(设置词汇字典),用于提取每一张图像的BOW特征描述
            self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.descriptor_extractor,flann)        
            self.bow_img_descriptor_extractor.setVocabulary(voc)   
            

    def predict(self,img):
        '''
        进行预测样本   
        load
        args:
            img：图像数据 
        args：
            label：样本所属类别标签，和训练输入标签值一致
            score：置信度 分数越低，置信度越高，表示属于该类的概率越大
            
        '''
        #转换为灰色
        if len(img.shape) == 3:
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            
        #提取图片的BOW特征描述
        #data = self.bow_descriptor_extractor(img_path)
        keypoints = self.feature_detector.detect(img)
        if  keypoints:
            data = self.bow_img_descriptor_extractor.compute(img,keypoints)
            _,result = self.svm.predict(data)
            #所属标签
            label = result[0][0]
            #设置标志位 获取预测的评分  分数越低，置信度越高，表示属于该类的概率越大
            a,res = self.svm.predict(data,flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)                
            score = res[0][0]
            #print('Label:{0}  Score：{1}'.format(label,score))            
            return label,score
        else:
            return None,None
        
            
    def sift_descriptor_extractor(self,img_path):
        '''
        特征提取：提取数据集中每幅图像的特征点，然后提取特征描述符，形成特征数据(如：SIFT或者SURF方法)；
        
        args：
            img_path：图像全路径
        '''        
        im = cv2.imread(img_path,0)
        keypoints = self.feature_detector.detect(im)
        if keypoints:
            return self.descriptor_extractor.compute(im,keypoints)[1]
        else:
            return None
    
    
    def bow_descriptor_extractor(self,img_path):
        '''
        提取图像的BOW特征描述(即利用视觉词袋量化图像特征)
        
        args：
            img_path：图像全路径
        '''        
        im = cv2.imread(img_path,0)
        keypoints = self.feature_detector.detect(im)
        if  keypoints:
            return self.bow_img_descriptor_extractor.compute(im,keypoints)
        else:
            return None