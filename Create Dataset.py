#Bismillah hirrahmaniraheem 786
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math


#receiving camera feed
cap = cv2.VideoCapture(0)

#initializing Detector Model
detector = HandDetector(maxHands=1)

folder = 'Data/Z'
counter = 0

offset = 20
imageSize = 300

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType = False)
    whiteImage = np.ones((imageSize, imageSize, 3), np.uint8) * 255



    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        #for showing only the hand detected
        cropImage = img[y-offset:y+h+offset, x-offset:x+w+offset]

        #Creating white image for fixed image size
        # whiteImage = np.ones((imageSize,imageSize, 3),np.uint8)*255


        aspectRatio = h/w

        if aspectRatio > 1:

            k = imageSize/h
            wCal = math.ceil(k * w)

            imageResize = cv2.resize(cropImage,(wCal,imageSize))

            #Calculating width gap to fix the image in center
            wGap  = math.ceil((300 - wCal)/2)

            #appending crop Image to white image
            whiteImage[:,wGap: wCal + wGap] = imageResize

        else:
            k = imageSize / w
            hCal = math.ceil(k * h)

            imageResize = cv2.resize(cropImage, (imageSize, hCal))

            # Calculating width gap to fix the image in center
            hGap = math.ceil((imageSize - hCal) / 2)

            # appending crop Image to white image
            whiteImage[hGap:hCal+hGap, :] = imageResize

            cv2.imshow('Only Hand', cropImage)
            cv2.imshow('White Image', whiteImage)


    # Applying Filter to White Image
    # Preprocessing Images:

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






    cv2.imshow('img',img)

    #New Lines For main.py
    cv2.imshow('grey',imggrey)
    cv2.imshow('blur',imgBlur)
    cv2.imshow('threshold',imgThresh)
    cv2.imshow('imgMedian',imgMedian)
    cv2.imshow('imgDilate',imgDilate)

    #saving Images to Respected Folders when "s" is pressed
    key = cv2.waitKey(1)

    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{counter}.jpg',imgDilate)
        print(counter)

