import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# initializes a video capture object
# argument `0` specifies default camera
cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1) # initializes a hand detector object

margin = 30
size = 300

counter = 0

while True:
    success, img = cap.read() # reads a frame `img` from the video capture object `cap`
    hands, img = detector.findHands(img) # returns a list of dictionary `hands` and `img` with drawings

    if hands:
        hand = hands[0] # accessing the first dictionary `hand`
        x, y, w, h = hand["bbox"] # accessing the `bbox` key

        # creates a numpy array representing an image of `size` by `size`
        # `3` indicates coloured
        # `np.uint8` specifies the datatype of the elements in the array
        # `* 255` to obtain white
        img_constant_size = np.ones((size, size, 3), np.uint8) * 255 # a white image of constant size

        img_crop = img[y - margin:y + h + margin, x - margin:x + w + margin] # a zoomed in image of the hand
        img_crop_shape = img_crop.shape # `img_crop.shape` returns an tuple with 3 values, height, width and colour channel of the image

        aspectRatio = h / w
        if aspectRatio > 1: # height of `hand` > width of `hand`
            multiplier = size / h
            stretched_width = math.ceil(multiplier * w)

            img_crop_resize = cv2.resize(img_crop, (stretched_width, size)) # resizing `img_crop` to have a fixed height of `size` and stretches width accordingly
            img_crop_resize_shape = img_crop_resize.shape
            img_crop_resize_shape_height = img_crop_resize_shape[0]
            img_crop_resize_shape_width = img_crop_resize_shape[1]

            right_shift_amount = math.ceil((size - img_crop_resize_shape_width) / 2) # the amount to shift img_crop_resize by to be able to center it

            # `img_constant_size[start_height:end_height, start_width:end_width]`
            img_constant_size[0:img_crop_resize_shape_height, right_shift_amount:img_crop_resize_shape_width + right_shift_amount] = img_crop_resize # overlays `img_crop` on top of `img_constant_size`
        elif aspectRatio < 1: # width of `hand` > height of `hand`
            multiplier = size / w
            stretched_height = math.ceil(multiplier * h)

            img_crop_resize = cv2.resize(img_crop, (size, stretched_height))
            img_crop_resize_shape = img_crop_resize.shape
            img_crop_resize_shape_height = img_crop_resize_shape[0]
            img_crop_resize_shape_width = img_crop_resize_shape[1]

            up_shift_amount = math.ceil((size - img_crop_resize_shape_height) / 2)

            img_constant_size[up_shift_amount:img_crop_resize_shape_height + up_shift_amount, 0:img_crop_resize_shape_width] = img_crop_resize
                    
        cv2.imshow("ImageCrop", img_crop)
        cv2.imshow("ImageConstantSize", img_constant_size)

    cv2.imshow("Image", img) # displays the frame `img` in a window of name `Image`
    
    # For Data Collection
    key = cv2.waitKey(1)
    file_path = "data/z"
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{file_path}/image_{time.time()}.jpg', img_constant_size)
        print(counter)




# https://github.com/cvzone/cvzone/blob/master/cvzone/HandTrackingModule.py