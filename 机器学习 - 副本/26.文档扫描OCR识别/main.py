import numpy as np
import argparse
import cv2

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation=inter)
	return resized


image = cv2.imread(r'E:/gupao/项目实战/机器学习/26.文档扫描OCR识别/img.png')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = resize(orig, height = 500)





















