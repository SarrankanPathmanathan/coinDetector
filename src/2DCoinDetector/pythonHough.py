"""
coin detection using opencv and python
task: draw a circle around each coin in an image
method:
- find the edge using Gaussian and Canny
- try to fit a circle to the edges by comparing circles of increasing size. Once passed threshold,
  assume that is the radius of coin and save the coordinates of the center
- draw the circles to the original image
"""

import cv2
import numpy as np
import math

coins = cv2.imread('../TestSet/testblack.png', 1)
cv2.imshow("Original", coins)
shifted = cv2.pyrMeanShiftFiltering(coins, 40, 80)
cv2.imshow("PyramidMeanShiftFilter", shifted)
# convert the mean shift image to grayscale, then apply
# Otsu's thresholding
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
thresh = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)

dp = 1
minDist = 30
param1 = 50
param2 = 30
min_r = 0
max_r = 50


def hough_circle_detection():
    img = thresh
    circles = cv2.HoughCircles(
        img,  # source imagef
        cv2.HOUGH_GRADIENT,  # type of detection
        dp,
        minDist,
        param1=param1,
        param2=param2,
        minRadius=min_r * 2,  # minimal radius
        maxRadius=max_r * 2,  # max radius
    )

    coins_copy = coins.copy()

    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle
        detected_radius = round(detected_radius)
        print(detected_radius)
        print(len(circles))
        coins_detected = cv2.circle(coins_copy, (x_coor, y_coor), detected_radius * 1, (0, 0, 255), 2)

    cv2.imshow("Output", coins_detected)
    cv2.waitKey(0)


def compare_circle_detection():
    hough_circle_detection()


compare_circle_detection()
