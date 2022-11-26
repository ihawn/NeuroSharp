import math

from PIL import Image
import numpy as np

#one line of input only (all pixel values on same line of file)
rawPixelList = []

with open("a.txt", "r") as filestream:
    for line in filestream:
        rawPixelList = line.split(",")

size = int(math.sqrt(len(rawPixelList)))

pixelList = []
for y in range(size):
    row = []
    for x in range(size):
        val = int(255 * float(rawPixelList[y * size + x]))
        row.append(val)
    pixelList.append(row)

pixelArray = np.array(pixelList)
new_image = Image.fromarray(pixelArray)
new_image.show()