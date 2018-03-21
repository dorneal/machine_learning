#!/usr/bin/python3
# coding:utf-8
# Filename:test_girl.py
# Author:黄鹏
# Time:2018.03.21 17:09

import numpy as np
import operator

"""
测试该女性属于哪一身材阶层（纯属搞笑，切勿当真）
"""


def create_data_set():
    """
    创建数据集
    :return: 数据集跟标签集
    """
    data_set = np.array(
        [[171, 85], [160, 107], [158, 106], [158, 90], [160, 86], [162, 84], [165, 88], [170, 96], [167, 90],
         [168, 94]])
    label_set = ['1分', '2分', '3分', '4分', '5分', '6分', '7分', '8分', '9分', '10分']
    return data_set, label_set


def classify(in_x, data_set, labels, k):
    """
    KNN算法分类器
    :param in_x: 用于分类的数据(测试集)
    :param data_set: 用于训练的数据(训练集)
    :param labels: 分类标签
    :param k: KNN算法参数，选择距离最小的K个点
    :return: 分类结果
    """
    data_set_size = data_set.shape[0]
    df = np.tile(in_x, (data_set_size, 1))
    diff_mat = df - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distance = sq_diff_mat.sum(axis=1)
    distances = sq_distance ** 0.5
    sorted_by_index_list = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_by_index_list[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


if __name__ == '__main__':
    group, labels = create_data_set()
    test = [173, 108]
    test_class = classify(test, group, labels, 3)
    print(test_class)
