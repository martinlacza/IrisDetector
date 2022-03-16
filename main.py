import cv2 as cv
import pandas as pd
import numpy as np


#df = pd.read_csv('data/duhovka.csv')

#for index, row in df.iterrows():
#    img = cv.imread('data/' + row['nazov'])
#    cv.circle(img, (row['zx'], row['zy']),row['zp'], (0,255,0), 2)
#    cv.circle(img, (row['dx'], row['dy']), row['dp'], (0, 0, 255), 2)
#    cv.imshow(row['nazov'], img)
#    cv.waitKey(0)
def on_trackbar(val):
    sigma = (cv.getTrackbarPos('Sigma', title_window) / 10)
    kernel_size = cv.getTrackbarPos('Kernel Size', title_window)
    if kernel_size == 0:
        kernel_size += 3
    if kernel_size % 2 == 0:
        kernel_size +=1
    kenny_low_threshold = cv.getTrackbarPos('Canny Lower Threshold', title_window)
    kenny_upper_threshold = cv.getTrackbarPos('Canny Upper Threshold', title_window)
    blurTrackbar = cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    canny = cv.Canny(blurTrackbar, kenny_low_threshold, kenny_upper_threshold)
    cv.imshow(title_window, canny)

def hough_trackbar(val):
    output = cv.imread('data/eye2.bmp')
    sigma =(cv.getTrackbarPos('Sigma', hough_window) / 10)
    kernel_size = cv.getTrackbarPos('Kernel Size', hough_window)
    if kernel_size == 0:
        kernel_size += 3
    if kernel_size % 2 == 0:
        kernel_size +=1
    print('sigma and kernel', sigma, kernel_size)
    resolution =  (cv.getTrackbarPos('Res', hough_window) / 10)
    mindist = cv.getTrackbarPos('Center', hough_window)
    minRad = cv.getTrackbarPos('MinRad', hough_window)
    maxRad = cv.getTrackbarPos('MaxRad', hough_window)
    kenny_low_threshold = cv.getTrackbarPos('CannyL', hough_window)
    kenny_upper_threshold = cv.getTrackbarPos('CannyU', hough_window)
    blurTrackbar = cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    hough_cirlces = cv.HoughCircles(blurTrackbar, cv.HOUGH_GRADIENT, resolution, mindist, param1=kenny_low_threshold,param2=kenny_upper_threshold, minRadius=minRad, maxRadius=maxRad )
    detected_circles = np.uint16(np.around(hough_cirlces))
    print(str(detected_circles))
    for (x, y, r) in detected_circles[0, :]:
        cv.circle(output, (x,y), r, (0,255,0),3)
        cv.circle(output, (x, y), 2, (0, 0, 255), 3)
    cv.imshow(hough_window, output)



kernel_slider_max = 50
canny_slider_max = 300
sigma_slider_max = 1000
title_window = 'Gaussian Blur Image'

image = cv.imread('data/eye2.bmp', cv.IMREAD_GRAYSCALE)

gblur = cv.GaussianBlur(image , (9,9),0.1)




cv.namedWindow(title_window)
cv.createTrackbar('Sigma', title_window , 0, sigma_slider_max, on_trackbar)
cv.createTrackbar('Kernel Size', title_window , 0, kernel_slider_max, on_trackbar)
cv.createTrackbar('Canny Lower Threshold', title_window , 0, canny_slider_max, on_trackbar)
cv.createTrackbar('Canny Upper Threshold', title_window , 0,canny_slider_max, on_trackbar)

hough_window = 'Hough Tramsformation'
cv.namedWindow('Hough Tramsformation')
cv.createTrackbar('Sigma', hough_window , 0, sigma_slider_max, hough_trackbar)
cv.createTrackbar('Kernel Size', hough_window , 0, kernel_slider_max, hough_trackbar)
cv.createTrackbar('Res', hough_window , 0, canny_slider_max, hough_trackbar)
cv.createTrackbar('Center', hough_window , 0, canny_slider_max, hough_trackbar)
cv.createTrackbar('MinRad', hough_window , 0, canny_slider_max, hough_trackbar)
cv.createTrackbar('MaxRad', hough_window, 0,canny_slider_max, hough_trackbar)
cv.createTrackbar('CannyL', hough_window , 0, canny_slider_max, hough_trackbar)
cv.createTrackbar('CannyU', hough_window, 0,canny_slider_max, hough_trackbar)

image1 = cv.imread('data/eye1.jpg', cv.IMREAD_GRAYSCALE)
image1copy = cv.imread('data/eye1.jpg')
cv.namedWindow('image1 Window')
sigma_image1= 2.1
kernel_size_image1 =11
res_image1=5.4
center_image1=1
minrad_image1=9
maxrad_image1=112
cannyL_image1=165
cannyU_image1=114
blur1 = cv.GaussianBlur(image1, (kernel_size_image1, kernel_size_image1), sigma_image1)
hough_cirlces_image1 = cv.HoughCircles(blur1, cv.HOUGH_GRADIENT, res_image1, center_image1, param1=cannyL_image1,param2=cannyU_image1, minRadius=minrad_image1, maxRadius=maxrad_image1 )
detected_circles_image1 = np.uint16(np.around(hough_cirlces_image1))
print(str(detected_circles_image1))
cv.circle(image1copy, (316,162),33,(0,0,255),3)
cv.circle(image1copy, (316,162),2,(0,0,255),3)
cv.circle(image1copy, (319,163),106, (0,0,255),3)
cv.circle(image1copy, (319,163),2,(0,0,255),3)
for (x, y, r) in detected_circles_image1[0, :]:
    cv.circle(image1copy, (x,y), r, (0,255,0),3)
    cv.circle(image1copy, (x, y), 2, (0, 0, 255), 3)
    cv.imshow('image1 Window', image1copy)


image2 = cv.imread('data/eye2.bmp', cv.IMREAD_GRAYSCALE)
image2copy = cv.imread('data/eye2.bmp')
cv.namedWindow('image2 Window')
sigma_image2= 7.8
kernel_size_image2 =15
res_image2=1
center_image2=6
minrad_image2=8
maxrad_image2=60
cannyL_image2=62
cannyU_image2=18
blur2 = cv.GaussianBlur(image2, (kernel_size_image2, kernel_size_image2), sigma_image2)
hough_cirlces_image2 = cv.HoughCircles(blur2, cv.HOUGH_GRADIENT, res_image2, center_image2, param1=cannyL_image2,param2=cannyU_image2, minRadius=minrad_image2, maxRadius=maxrad_image2 )
detected_circles_image2 = np.uint16(np.around(hough_cirlces_image2))
print(str(detected_circles_image2))
cv.circle(image2copy, (155,117),24,(0,0,255),3)
cv.circle(image2copy, (155,117),2,(0,0,255),3)
cv.circle(image2copy, (155,117),56, (0,0,255),3)
cv.circle(image2copy, (155,117),2,(0,0,255),3)
for (x, y, r) in detected_circles_image2[0, :]:
    cv.circle(image2copy, (x,y), r, (0,255,0),3)
    cv.circle(image2copy, (x, y), 2, (0, 0, 255), 3)
    cv.imshow('image2 Window', image2copy)


image3 = cv.imread('data/eye3.jpg', cv.IMREAD_GRAYSCALE)
image3copy = cv.imread('data/eye3.jpg')
cv.namedWindow('image3 Window')
sigma_image3= 33
kernel_size_image3 =27
res_image3=4.5
center_image3=5
minrad_image3=32
maxrad_image3=217
cannyL_image3=61
cannyU_image3=48
blur3 = cv.GaussianBlur(image3, (kernel_size_image3, kernel_size_image3), sigma_image3)
hough_cirlces_image3 = cv.HoughCircles(blur3, cv.HOUGH_GRADIENT, res_image3, center_image3, param1=cannyL_image3,param2=cannyU_image3, minRadius=minrad_image3, maxRadius=maxrad_image3 )
detected_circles_image3 = np.uint16(np.around(hough_cirlces_image3))
print(str(detected_circles_image3))
cv.circle(image3copy, (234,230),56,(0,0,255),3)
cv.circle(image3copy, (234,230),2,(0,0,255),3)
cv.circle(image3copy, (248,232),221, (0,0,255),3)
cv.circle(image3copy, (248,232),2,(0,0,255),3)
for (x, y, r) in detected_circles_image3[0, :]:
    cv.circle(image3copy, (x,y), r, (0,255,0),3)
    cv.circle(image3copy, (x, y), 2, (0, 255, 0), 3)
    cv.imshow('image3 Window', image3copy)

on_trackbar(0)

cv.waitKey(0)