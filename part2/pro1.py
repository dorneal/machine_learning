#!/usr/bin/python3
# coding:utf-8
# Filename:pro1.py
# Author:黄鹏
# Time:2018.03.21 10:40


def remove(keywords):
    new_list = []
    old_list = keywords.split('!')
    for i in reversed(old_list):
        if '' == i:
            new_list.append('!')
        else:
            break
    merge_list = old_list + new_list
    output_str = ''
    for this_str in merge_list:
        output_str += this_str
    return output_str


def remove2(key):
    suffix = ''
    for s in reversed(key):
        if s != '!':
            break
        suffix += '!'
    return key.replace('!', '') + suffix


def find_low(arr):
    min_num = arr[0]
    for i in range(len(arr)):
        if min_num > arr[i]:
            min_num = arr[i]
    return min_num


old_str = '!!jack!!! neal!!'
print("旧字符：" + old_str)
print("新字符：" + remove2(old_str))

test_list = [65, -65, -4, 8, 4, 321, 2, 4]
print(find_low(test_list))
