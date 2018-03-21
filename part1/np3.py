#!/usr/bin/python3
# coding:utf-8
# Filename:np3.py
# Author:黄鹏
# Time:2018.03.20 17:11
import operator
from math import log
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    # 分类属性
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    # 返回数据集跟分类属性
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算给定数据集的经验熵
    :param dataSet: 数据集
    :return: 经验熵（香农熵）
    """
    numEntires = len(dataSet)  # 返回数据集行数
    labelCounts = {}  # 保存每个标签(Label)出现次数的字典
    for featVec in dataSet:  # 对于每组特征向量进行统计
        currentLabel = featVec[-1]  # 提取标签(Label)信息
        if currentLabel not in labelCounts.keys():  # 如果标签(Label)没有放入统计次数
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # Label计数
    shannoEnt = 0.0  # 经验熵(香农熵)
    for key in labelCounts:  # 计算香农熵
        prob = float(labelCounts[key]) / numEntires  # 选择该标签的概率
        shannoEnt -= prob * log(prob, 2)  # 利用公式计算
    return shannoEnt  # 返回经验熵(香农熵)


def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet: 数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征值
    :return:
    """
    retDataSet = []  # 创建返回的数据集列表
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 去掉axis特征
            reducedFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet  # 返回划分后的数据集


def chooseBestFeatureToSplit(dataSet):
    """
    选择最优特征
    :param dataSet:数据集
    :return:信息增益最大的（最优）特征的索引值
    """
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
    bestInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优特征的索引值
    for i in range(numFeatures):  # 遍历所有特征
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建set集合{}，元素不可重复
        newEntroy = 0.0  # 经验条件熵
        for value in uniqueVals:  # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)  # subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntroy += prob * calcShannonEnt(subDataSet)  # 根据公式计算经验条件熵
        infoGain = baseEntropy - newEntroy  # 信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if infoGain > bestInfoGain:  # 计算信息增益
            bestInfoGain = infoGain  # 更新信息增益
            bestFeature = i  # 记录信息增益的最大特征的索引值
    return bestFeature  # 返回信息增益的最大特征的索引值


def majorityCnt(classList):
    """
    统计classList中出现此处最多的元素(类标签)
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:  # 统计classList中每个元素出现的次数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # 返回classList中出现次数最多的元素


def createTree(dataSet, labels, featLabels):
    """
    创建决策树
    :param dataSet: 数据集
    :param labels: 标签
    :param featLabels:
    :return:
    """
    classList = [example[-1] for example in dataSet]  # 取分类标签(是否放贷)
    if classList.count(classList[0]) == len(classList):  # 如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征
    bestFeatLabel = labels[bestFeat]  # 最优特征标签
    featLabels.append(bestFeatLabel)
    my_tree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树
    del (labels[bestFeat])  # 删除已经使用的特征标签
    featValues = [example[bestFeat] for example in dataSet]  # 得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)  # 去掉重复的属性值
    for value in uniqueVals:  # 遍历特征，创建决策树
        my_tree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return my_tree


def getNumLeafs(myTree):
    """
    获取决策树叶子节点的数目
    :param myTree:决策树
    :return: 决策树的叶子节点的数目
    """
    numLeafs = 0  # 初始化叶子
    firstStr = next(iter(myTree))  # python3中的myTree.keys()返回的是dict_keys,所以不能使员工myTree.keys()[0]方法获取
    secondDict = myTree[firstStr]  # 获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该节点是否为字典，如果不是字典，代表此节点为叶子节点
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """
    获取决策树的层数
    :param myTree: 决策树
    :return: 决策树的层数
    """
    maxDepth = 0  # 初始化决策树深度
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]  # 获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该节点是否为字段是否为字段，如果不是字典，则代表该节点是叶子节点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth  # 更新层数
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制节点
    :param nodeTxt: 节点名
    :param centerPt: 文本位置
    :param parentPt: 标注的箭头位置
    :param nodeType: 节点格式
    :return:
    """
    arrow_args = dict(arrowstyle='<-')  # 定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 设置字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt,
                            textcoords='axes fraction', va='center', ha='center',
                            bbox=nodeType, arrowprops=arrow_args, FontProperties=font)  # 绘制节点


def plotMidText(cntrPt, parentPt, txtString):
    """
    标注有向边属性值
    :param cntrPt:标注的位置
    :param parentPt: 标注的位置
    :param txtString: 标注的内容
    :return:
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算标注位置
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va='center', ha="center",
                        rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    """
    绘制决策树
    :param myTree: 决策树（字典）
    :param parentPt: 标注的内容
    :param nodeTxt: 节点名
    :return:
    """
    decisionNode = dict(boxstyle='sawtooth', fc="0.8")  # 设置节点格式
    leafNode = dict(boxstyle='round4', fc="0.8")  # 设置叶节点格式
    numLeafs = getNumLeafs(myTree)  # 获取决策树叶节点数目，决定了树的宽度
    depth = getTreeDepth(myTree)  # 获取决策树层度
    firstStr = next(iter(myTree))  # 下一个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制节点
    secondDict = myTree[firstStr]  # 下一个字典，也就是继续绘制子节点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))  # 不是叶节点，递归继续绘制
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW  # 是叶节点，绘制叶节点，并标注有向边属性值
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), "")
    plt.show()


def classsify(inputTree, featLabels, testVec):
    classLabel = None
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classsify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    # print(myTree)
    # createPlot(myTree)
    testVec = [0, 1]
    result = classsify(myTree, featLabels, testVec)
    if result == 'yes':
        print('放贷')
    elif result == 'no':
        print('不放贷')
