import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def FLD(image):
    # Create default Fast Line Detector class
    fld = cv.ximgproc.createFastLineDetector(length_threshold = 30, do_merge = True)
    # Get line vectors from the image
    lines = fld.detect(image)
    # Draw lines on the image
    line_on_image = fld.drawSegments(image, lines)
    # Plot
    # plt.imshow(line_on_image, interpolation='nearest', aspect='auto')
    # plt.show()
    return line_on_image

img = cv.imread('1.png')
assert img is not None, "file could not be read, check with os.path.exists()"

# histr = cv.calcHist([img],[0],None,[256],[0,256])
# plt.plot(histr,color = 'b')
# plt.xlim([0,256])
# plt.show()

a = np.where(img[:,:,0]>45, 0, img[:,:,0])


cv.imshow(' ', FLD(a))
cv.waitKey(0)