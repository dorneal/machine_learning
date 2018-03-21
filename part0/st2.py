#!/usr/bin/python3
# coding:utf-8
# Filename:st2.py
# Author:黄鹏
# Time:2018.03.20 11:02

from requests import Request, Session

s = Session()
url = 'https://www.baidu.com'
data = {'name': 'jack'}
header = {'User-agent': 'python-requests/2.18.4'}
req = Request('GET', url, data=data, headers=header)

prepped = req.prepare()
resp = s.send(prepped,stream=stream,)