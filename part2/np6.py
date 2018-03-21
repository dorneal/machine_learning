#!/usr/bin/python3
# coding:utf-8
# Filename:np6.py
# Author:黄鹏
# Time:2018.03.21 14:51

from sklearn import tree

if __name__ == '__main__':
    fr = open('lenses.txt')
    lense = [inst.strip().split('\t') for inst in fr.readlines()]
    print(lense)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    clf = tree.DecisionTreeClassifier()
    lenses = clf.fit(lense, lensesLabels)
