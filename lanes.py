import cv2
import numpy as np 
import matplotlib.pyplot as plt
# Detecting Edges of Lane Lines
# fork from https://github.com/rslim087a/road-video
def canny(image):
	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	#对图片应用高斯滤镜
	blur = cv2.GaussianBlur(gray,(5,5),0)
	canny = cv2.Canny(blur,50,150)
	return canny
# https://towardsdatascience.com/deeppicar-part-4-lane-following-via-opencv-737dd9e47c96
def detect_edges(frame):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    
    lower_blue = np.array([60, 40, 40])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # detect edges
    edges = cv2.Canny(mask, 200, 400)
    return edges

def region_of_interest(image):
	height = image.shape[0]
	polygons = np.array([
	[(200,height),(1100,height),(550,250)]
	])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask,polygons,255)
	maksed_image = cv2.bitwise_and(image,mask)
	return maksed_image

def display_lines(image,lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			#将line多维list变成一位list
			x1,y1,x2,y2 = line.reshape(4)
			#在line_image画布上绘制直线
			cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
	return line_image

def make_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	# print(image.shape)
	y1 = image.shape[0]
	y2 = int(y1*(3/5))
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
	left_fit = []
	right_fit = []
	for line in lines:
		x1,y1,x2,y2 = line.reshape(4)
		parameters = np.polyfit((x1 ,x2),(y1 ,y2),1)
		# print(parameters)
		slope =parameters[0]
		intercept =parameters[1]
		if slope < 0:
			left_fit.append((slope, intercept))
		else:
			right_fit.append((slope, intercept))
		# print(left_fit)
		# print(right_fit)
	left_fit_average = np.average(left_fit, axis = 0)
	right_fit_average = np.average(right_fit, axis = 0)
	# print(left_fit_average, "left")
	# print(right_fit_average, "right")
	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)
	return np.array([left_line, right_line])




# #读取图片
# image = cv2.imread('test_image.jpg')
# #复制图片
# lane_image = np.copy(image)
# #将彩色图片转换成灰度格式
# canny_image = canny(lane_image)
# #显示图片
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# average_lines = average_slope_intercept(lane_image,lines)
# line_image = display_lines(lane_image, average_lines)
# # 车道图像与原始图像混合
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image ,1 ,1)
# cv2.imshow('result',combo_image)
# # cv2.imshow('result',detect_edges(lane_image))
# cv2.waitKey(0)
# # plt.imshow('result',region_of_interest(canny))
# # plt.show()
# #
cap = cv2.VideoCapture("test2.mp4")
while (cap.isOpened()):
	_, frame = cap.read()
	#将彩色图片转换成灰度格式
	canny_image = canny(frame)
	#显示图片
	cropped_image = region_of_interest(canny_image)
	lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
	average_lines = average_slope_intercept(frame,lines)
	line_image = display_lines(frame, average_lines)
	# 车道图像与原始图像混合
	combo_image = cv2.addWeighted(frame, 0.8, line_image ,1 ,1)
	cv2.imshow('result',combo_image)
	cv2.waitKey(1) 