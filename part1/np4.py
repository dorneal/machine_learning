#!/usr/bin/python3
# coding:utf-8
# Filename:np4.py
# Author:黄鹏
# Time:2018.03.21 9:16


import pickle


def storeTree(inputTree, filename):
    """
    存储决策树
    :param inputTree: 已经生成的决策树
    :param filename: 决策树的存储名称
    :return:
    """
    with open(filename, 'wb') as f:
        pickle.dump(inputTree, f)


def grabTree(filename):
    """
    读取决策树
    :param filename: 文件名
    :return:
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    myTree = {'有自己的的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
    storeTree(myTree, 'classifierStorage.txt')
