import cv2
import numpy as np

# read original image
hubble_image = cv2.imread('Hubble-stars.jpg')
# cutout image
cutout_image = cv2.imread('cutout.jpg', 0)
# get cutout image size
cutout_width, cutout_height = cutout_image.shape[::-1]
# converting orifinal image to grayscale
bw_image = cv2.cvtColor(hubble_image, cv2.COLOR_BGR2GRAY)

matches = cv2.matchTemplate(bw_image, cutout_image, cv2.TM_CCOEFF_NORMED)
# setting reliability value
relaibility = 0.6
# posiitons of possible similair to cutout images
sim_pos = np.where(matches >= relaibility)
for position in zip(*sim_pos[::-1]):
    cv2.rectangle(hubble_image, position,
                  (position[0]+cutout_width, position[1]+cutout_height),
                  (0, 255, 0), 1)
# writing image of postions labeled with yellow boxes
cv2.imwrite('regions.jpg', hubble_image, params=None)
