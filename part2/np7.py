#!/usr/bin/python3
# coding:utf-8
# Filename:np7.py
# Author:黄鹏
# Time:2018.03.21 15:46
import numpy as np


def loadDataSet():
    """
    创建试验样本
    :return:
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def setOfWords2Vec(vocabList, inputSet):
    """
    根据VocabList词汇表，将inputSet向量化，向量的每个元素为1或者0
    :param vocabList: createVocabList返回的列表
    :param inputSet: 切分的词条列表
    :return:  文档向量，词集模型
    """
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:  # 遍历每个词条
        if word in vocabList:  # 如果词条存在词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary!" % word)
    return returnVec  # 返回文档向量


def createVocabList(dataSet):
    """
    将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
    :param dataSet: 整理的样本数据集
    :return: 返回不重复的词条列表，也就是词汇表
    """
    vocabSet = set([])  # 创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)


def trainNBO(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix:  训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    :param trainCategory: 训练类别标签向量，即loadDataSet返回的classVec
    :return:
    """
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于侮辱类的概率
    p0Num = np.zeros(numWords)  # 创建numpy.zeros数组，词条出现数初始化为0
    p1Num = np.zeros(numWords)
    p0Denom = 0.0  # 分母初始化为0
    p1Denom = 0.0
    for i in range(numTrainDocs):  # 统计输入侮辱类的条件概率
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 统属于非侮辱类的条件概率
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive  # 返回属于侮辱类的条件概率


if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    print('myVocabList:\n', myVocabList)
    trainMat = []
    for postingDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postingDoc))
    p0V, p1V, pAb = trainNBO(trainMat, classVec)
    print('p0V:\n', p0V)
    print('p1V:\n', p1V)
    print("classVec:\n", classVec)
    print("pAb:\n", pAb)
