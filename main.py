import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)

classifier = Classifier("model/keras_model.h5", "model/labels.txt")

margin = 30
size = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

counter = 0

while True:
    success, img = cap.read()
    img_output = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        img_constant_size = np.ones((size, size, 3), np.uint8) * 255

        img_crop = img[y - margin:y + h + margin, x - margin:x + w + margin]
        img_crop_shape = img_crop.shape

        aspectRatio = h / w
        if aspectRatio > 1:
            multiplier = size / h
            stretched_width = math.ceil(multiplier * w)

            img_crop_resize = cv2.resize(img_crop, (stretched_width, size)) 
            img_crop_resize_shape = img_crop_resize.shape
            img_crop_resize_shape_height = img_crop_resize_shape[0]
            img_crop_resize_shape_width = img_crop_resize_shape[1]

            right_shift_amount = math.ceil((size - img_crop_resize_shape_width) / 2)

            # `img_constant_size[start_height:end_height, start_width:end_width]`
            img_constant_size[0:img_crop_resize_shape_height, right_shift_amount:img_crop_resize_shape_width + right_shift_amount] = img_crop_resize

            prediction, index = classifier.getPrediction(img_constant_size, draw=False)
        elif aspectRatio < 1:
            multiplier = size / w
            stretched_height = math.ceil(multiplier * h)

            img_crop_resize = cv2.resize(img_crop, (size, stretched_height))
            img_crop_resize_shape = img_crop_resize.shape
            img_crop_resize_shape_height = img_crop_resize_shape[0]
            img_crop_resize_shape_width = img_crop_resize_shape[1]

            up_shift_amount = math.ceil((size - img_crop_resize_shape_height) / 2)

            img_constant_size[up_shift_amount:img_crop_resize_shape_height + up_shift_amount, 0:img_crop_resize_shape_width] = img_crop_resize
            
            prediction, index = classifier.getPrediction(img_constant_size, draw=False)
        
        cv2.putText(img_output, labels[index], (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(img_output, (x - margin, y - margin), (x + w + margin, y + h + margin), (255, 0, 255), 5)
        
        cv2.imshow("ImageCrop", img_crop)
        cv2.imshow("ImageConstantSize", img_constant_size)
    
    cv2.imshow("Image", img_output)
    cv2.waitKey(1)