import cv2
import glob

def readImage(fileName):
    img = cv2.imread(fileName, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    return img

def filesFromFolder(dir):
    print (dir + "*.jpg")
    return glob.glob(dir + "*.jpg")

def getImages(dir):
    newImageList = []
    images = filesFromFolder(dir)

    for image in images:
        print (image)
        image = (readImage(image))

        rows,cols = image.shape

        smallGrey = cv2.resize(image,(32,32))

        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        dst = cv2.warpAffine(image,M,(cols,rows))

        newImageList.append(dst)

    return newImageList, images

def rotate(image, degree):
    # print (image)
    # for image_index in range(len(image_list)):
    # image = image_list[image_index]
    rows,cols = image.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),int(degree),1)
    dst = cv2.warpAffine(image,M,(cols,rows))

    image = dst

    return image

def saveImage(directory, image):
    cv2.imwrite(directory,image)
