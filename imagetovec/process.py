import image_utils
import cv2

imageList, imageNames = image_utils.getImages("../randomimages/")

for x in range(len(imageList)):
    image = imageList[x]
    rows,cols = image.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    dst = cv2.warpAffine(image,M,(cols,rows))
    imageList[x] = dst

image_utils.saveImage(imageList[0], "dir.jpg")

rowList = []


with tf.variable_scope
for x in imageList():
    for row in imageList.shape[0]:?
        if(rowList.append())
