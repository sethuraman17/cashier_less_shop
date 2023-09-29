import cv2
import cvzone

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

while True:
    success, img1 = cap1.read()
    ret, img2 = cap2.read()
    cv2.imshow("Images", img1)
    cv2.imshow("Image", img2)
    cv2.waitKey(1)
