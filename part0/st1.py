#!/usr/bin/python3
# coding:utf-8
# Filename:st1.py
# Author:黄鹏
# Time:2018.03.20 10:39

import requests

r = requests.get('https://github.com/timeline.json')
print(r.headers)
print(r.status_code)
if r.status_code == requests.codes.ok:
    print(r.json())
else:
    print(r.request.headers)