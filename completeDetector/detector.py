# import the necessary packages

import tensorflow.keras
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.segmentation import watershed
from PIL import Image, ImageOps
from scipy import ndimage
import numpy as np
import imutils
import cv2
import os
import shutil
import time
import decimal

TWOPLACES = decimal.Decimal(10) ** -2


def getValue(index):
    if index == 0:
        return 0.01
    elif index == 1:
        return 0.05
    elif index == 2:
        return 0.10
    elif index == 3:
        return 0.25
    elif index == 4:
        return 0.50
    elif index == 5:
        return 1.00
    elif index == 6:
        return 2.00


def getPrediction(img):
    # Replace this with the path to your image

    # Load the model
    image = Image.open(img)

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    # image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print("Prediction Result for " + img)
    print(" Penny : " + str(prediction[0][0]) + "\n Nickel : " + str(prediction[0][1]) + "\n Dime : " + str(
        prediction[0][2]) + "\n Quarter : " + str(prediction[0][3]) + "\n Loonie : " + str(
        prediction[0][4]) + "\n Toonie : " + str(prediction[0][5]))
    print(max(prediction[0]))
    if max(prediction[0]) >= 0.50:
        result = max(prediction[0])
        index = np.where(prediction == result)[1][0]
    else:
        index = -1
    print("Returned index=" + str(index))
    return index


def cleanDirectory(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def preProcessImageHough(image, debug):
    shifted = cv2.pyrMeanShiftFiltering(image, 10, 30)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 13)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if debug:
        cv2.imshow("Original", image)
        cv2.imshow("Blur", shifted)
        cv2.imshow("Gray", gray)
        cv2.imshow("Thresh", thresh)
        cv2.waitKey(0)
    return thresh


def preProcessImageWatershed(image, debug):
    shifted = cv2.pyrMeanShiftFiltering(image, 40,80)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if debug:
        cv2.imshow("Original", image)
        cv2.imshow("PyramidMeanShiftFilter", shifted)
        cv2.imshow("Gray", gray)
        cv2.imshow("Thresh", thresh)
        cv2.waitKey(0)
    return thresh


def waterShedCoinDetection(image, debug, getSum):
    img = preProcessImageWatershed(image, debug)
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then apply the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    # loop over the unique labels returned by the Watershed
    # algorithm
    circles = []
    radius = []
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(img.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        circles.append(c)
        radius.append(r)

    image_copy = image.copy()

    thresh = 2
    goodRadius = []
    outlier = 0
    outliers = [(t > thresh) for t in z_score(radius)]
    print(str(min(radius)))
    print(str(max(radius) - min(radius)))
    for i, r in enumerate(radius):
        if not outliers[i]:
            if (abs(r-max(radius))  < 79 and r > 10):
                goodRadius.append(r)
            else:
                outlier = outlier + 1
        else:
            outlier = outlier +1
    print("Number of Outlier:" + str(outlier))
    print(len(goodRadius))


    goodCircles = []
    for detected_circle in circles:
        ((x, y), r) = cv2.minEnclosingCircle(detected_circle)
        if r in goodRadius:
            goodCircles.append(detected_circle)

    ROI_NUMBER = 0
    ROI = []
    sum = 0
    print(len(goodCircles))
    print(len(circles))
    for circle in goodCircles:
        ((x, y), r) = cv2.minEnclosingCircle(circle)
        x1 = int(x - (r - 4))
        y1 = int(y + (r - 4))
        x2 = int(x + (r - 4))
        y2 = int(y - (r - 4))

        if getSum:
            Predictions = image[y2:y1, x1:x2]
            cv2.imwrite('./Predictions/ROI_{}.png'.format(ROI_NUMBER), Predictions)
            ROI.append('./Predictions/ROI_{}.png'.format(ROI_NUMBER))
            cv2.imwrite('./Predictions/ROI_{}.png'.format(ROI_NUMBER), Predictions)
            time.sleep(1)
            index = getPrediction('./Predictions/ROI_{}.png'.format(ROI_NUMBER))
            if index != -1:
                print("FOUND:" + str(coinsValues[index]))
                sum = sum + coinsValues[index][1]
                print(ROI_NUMBER)
            else:
                print("FOUND NO MATCHES")

        start_point = (x1, y1)
        end_point = (x2, y2)
        color = (0, 0, 255)
        thickness = 2
        # image = cv2.rectangle(image_copy, start_point, end_point, color, thickness)
        cv2.circle(image_copy, (int(x), int(y)), int(r), (0, 255, 0), 2)
        # cv2.putText(image_copy, "${}".format(str(coinsValues[index][1])), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        ROI_NUMBER += 1
    # cv2.putText(image_copy, "The estimated total is $" + str(decimal.Decimal(sum).quantize(TWOPLACES)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Output", image_copy)
    cv2.waitKey(0)
    return sum


def z_score(ys):
    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = np.abs([(y - mean_y) / stdev_y for y in ys])
    return z_scores


def hough_circle_detection(debug, getSum, img, dp, minDist, param1, param2, min_r, max_r):
    coins_copy = img.copy()
    processedImg = preProcessImageHough(img, debug)
    circles = cv2.HoughCircles(
        coins_copy,  # source imagef
        cv2.HOUGH_GRADIENT,  # type of detection
        dp,
        minDist,
        param1=param1,
        param2=param2,
        minRadius=min_r * 2,  # minimal radius
        maxRadius=max_r * 2,  # max radius
    )

    radius = []
    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle
        radius.append(detected_radius)

    thresh = 20
    goodRadius = []
    outliers = [(t > thresh) for t in z_score(radius)]
    print("Number of Outlier:" + str(len(outliers)))
    for i, r in enumerate(radius):
        if not outliers[i]:
            goodRadius.append(r)

    goodCircles = []
    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle
        if detected_radius in goodRadius:
            goodCircles.append(detected_circle)

    ROI_NUMBER = 0
    ROI = []
    image_copy = coins_copy.copy()
    sum = 0
    for detected_circle in goodCircles:
        x_coor, y_coor, detected_radius = detected_circle
        detected_radius = int(detected_radius)
        x1 = int(x_coor - (detected_radius - 13))
        y1 = int(y_coor + (detected_radius - 13))
        x2 = int(x_coor + (detected_radius - 13))
        y2 = int(y_coor - (detected_radius - 13))
        if getSum:
            Predictions = coins_copy[y2:y1, x1:x2]
            cv2.imwrite('./Predictions/ROI_{}.png'.format(ROI_NUMBER), Predictions)
            ROI.append('./Predictions/ROI_{}.png'.format(ROI_NUMBER))
            cv2.imwrite('./Predictions/ROI_{}.png'.format(ROI_NUMBER), Predictions)
            time.sleep(1)
            index = getPrediction('./Predictions/ROI_{}.png'.format(ROI_NUMBER))
            if index != -1:
                print("FOUND:" + str(coinsValues[index]))
                sum = sum + coinsValues[index][1]
                print(ROI_NUMBER)
            else:
                print("FOUND NO MATCHES")
            ROI_NUMBER += 1
        start_point = (x1, y1)
        end_point = (x2, y2)
        color = (0, 0, 255)
        thickness = 2
        coins_detected = cv2.rectangle(image_copy, start_point, end_point, color, thickness)
        cv2.putText(image_copy, "${}".format(str(coinsValues[index][1])), (int(x_coor) - 10, int(y_coor)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        coins_detected = cv2.circle(image_copy, (int(x_coor), int(y_coor)), int(detected_radius - 13), (0, 255, 0), 2)
    cv2.putText(image_copy, "The estimated total is $" + str(decimal.Decimal(sum).quantize(TWOPLACES)),
     (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Detected", coins_detected)
    cv2.waitKey(0)
    return sum


test_dir = './TestSet/'
coinsValues = [["Penny", 0.01], ["Nickel", 0.05], ["Dime", 0.10], ["Quarter", 0.25], ["Loonie", 1.00], ["Toonie", 2.00]]
model = tensorflow.keras.models.load_model('keras_model.h5')


def main():
    cleanDirectory("./Predictions")
    for filename in os.listdir(test_dir):
        coinPath = os.path.join(test_dir, filename)
        print(coinPath)
        coins = cv2.imread(coinPath, 1)
        #ROI_Watershed = waterShedCoinDetection(coins, False, False)
        ROI_Hough = hough_circle_detection(False, False, coins, dp=1, minDist=60, param1=40, param2=20, min_r=15, max_r=50)


if __name__ == '__main__':
    main()
