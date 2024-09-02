import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
# from h5py.h5pl import append
import cvzone
import string

#receiving camera feed
cap = cv2.VideoCapture(0)

#initializing Detector Model
detector = HandDetector(maxHands=1)
classifier = Classifier('Model/A-G/keras_model.h5','Model/A-G/labels.txt')

# labels = ['A','B','C']
# labels = list(string.ascii_uppercase)
all_letters = string.ascii_uppercase
labels = list(all_letters[:7])
all_labels = []
unique_labels = []
final_label = ''



offset = 20
imageSize = 300

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgOutput = img.copy()
    hands, img = detector.findHands(img,flipType=False)


    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        #for showing only the hand detected
        cropImage = img[y-offset:y+h+offset, x-offset:x+w+offset]

        #Creating white image for fixed image size
        whiteImage = np.ones((imageSize,imageSize, 3),np.uint8)*255


        aspectRatio = h/w

        if aspectRatio > 1:

            k = imageSize/h
            wCal = math.ceil(k * w)

            imageResize = cv2.resize(cropImage,(wCal,imageSize))

            #Calculating width gap to fix the image in center
            wGap  = math.ceil((300 - wCal)/2)

            #appending crop Image to white image
            whiteImage[:,wGap: wCal + wGap] = imageResize

            # prediction,index = classifier.getPrediction(whiteImage)
            # print(prediction, index)

        else:
            k = imageSize / w
            hCal = math.ceil(k * h)

            imageResize = cv2.resize(cropImage, (hCal, imageSize))

            # Calculating width gap to fix the image in center
            hGap = math.ceil((300 - hCal) / 2)

            # appending crop Image to white image
            whiteImage[:, hGap: hCal + hGap] = imageResize

        imggrey = cv2.cvtColor(whiteImage, cv2.COLOR_BGR2GRAY)

        # for reducing image noise and detail
        imgBlur = cv2.GaussianBlur(imggrey, (3, 3), 1)

        # for converting image to binary image with black and white pixels only
        imgThresh = cv2.adaptiveThreshold(imggrey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25,
                                          16)

        # for reducing image noice by removing extra white pixels
        imgMedian = cv2.medianBlur(imgThresh, 5)

        # optional For making the white pixel thicker
        kernel = np.ones((2, 2), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, 1)

        imgDilateRGB = cv2.cvtColor(imgDilate, cv2.COLOR_GRAY2RGB)

        prediction, index = classifier.getPrediction(imgDilateRGB, draw=False)
        print(prediction, index)

        all_labels.append(labels[index])

        # Displaying only unique labels
        unique_labels = []
        for i in range(len(all_labels) - 1):
            if all_labels[i] != all_labels[i + 1]:
                unique_labels.append(all_labels[i])

        # Ensure the last label is always included
        if all_labels:
            unique_labels.append(all_labels[-1])
            final_label = all_labels[-1]

        # Display all unique collected labels using cvzone.putTextRect
        cvzone.putTextRect(imgOutput, ''.join(unique_labels), (50, 50), scale=3, thickness=5, offset=20,
                           colorR=(0, 200, 0))

        #Rectangle Around Label
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)

        #Writing Lable
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

        #Rectangle Around Complete hand
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # cv2.imshow('Final Image', imgOutput)
        # cv2.imshow('Only Hand',cropImage)
        # cv2.imshow('White Image',whiteImage)
    else:
       unique_labels = []
       all_labels = []

    cv2.imshow('Final Image', imgOutput)
    # cv2.imshow('img',img)

    #saving Images to Respected Folders when "s" is pressed
    cv2.waitKey(1)


