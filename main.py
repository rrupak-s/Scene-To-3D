import numpy as np
import cv2 as cv
import matplotlib
# matplotlib.use('Qt5Agg')  # For Qt5Agg (requires PyQt5 installed)
# OR
matplotlib.use('Agg')  # For non-interactive rendering
import matplotlib.pyplot as plt
 
img1 = cv.imread('dataset/IMG_20241126_165245.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('dataset/IMG_20241126_165248.jpg', cv.IMREAD_GRAYSCALE)
# --------------------feature detection------------------
# Initiate ORB detector
orb = cv.ORB_create()
 
# # find the keypoints with ORB
# kp = orb.detect(img1,None)
 
# # compute the descriptors with ORB
# kp, des = orb.compute(img1, kp)
 
# draw only keypoints location,not size and orientation
# img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()
# plt.imshow(img2)
# plt.savefig("output_image.png")  # Saves the image to a file instead of displaying it

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


#--------------------feature matching--------------------

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
 
# Match descriptors.
matches = bf.match(des1,des2)
 
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
 
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
plt.imshow(img3)
plt.savefig("output_image.png") 

