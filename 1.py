
# from PIL import Image,ImageOps, ImageDraw, ImageFont, ImageFilter
import numpy as np
# image=Image.open("image.jpg")

# # image.show()
# import matplotlib.pyplot as plt
# plt.imshow(image)
# plt.show()
# image = Image.open("1.jpg").convert("L")

# colored = ImageOps.colorize(image, black="blue", white="yellow")
# colored.show()
# image=image.convert("RGB")
# r, g, b = image.split()
# image.show()
# image=ImageOps.grayscale(image)
# image1=image.quantize(18) #reduces the number of colors in an image.
# image1.show()
# image=ImageOps.flip(image)
# image=ImageOps.mirror(image)
# image.show()
# left = 50
# upper = 50
# right = 200
# lower = 200
# image_draw=image.copy()
# imge=ImageDraw.Draw(image_draw)
# shapes=[left,upper,right,lower]
# imge.rectangle(shapes,fill="blue")
# image_draw.show()
# center_x = 150
# center_y = 150
# radius = 80

# left = center_x - radius
# top = center_y - radius
# right = center_x + radius
# bottom = center_y + radius

# imge.ellipse([left, top, right, bottom], fill="blue")
# font = ImageFont.truetype("arial.ttf", 40)
# imge.text(xy=(0,0), text="Hello!", fill="red", font=font)
# image_draw.show()
# kernel = np.ones((5,5))/36
# image=image.filter(ImageFilter.Kernel((5,5),kernel.flatten()))
# image.show()


import cv2
import numpy as mp
image=cv2.imread("image.jpg")
# image=cv2.flip(image,0)
# cv2.imshow("Flipped Image", image)
# image=cv2.rotate(image,cv2.ROTATE_90_ANTICLOCKWISE)
# image=cv2.imread("image.jpg",cv2.IMREAD_GRAYSCALE)
# cv2.imshow("Grayscale Image", image)

# upper = 100
# lower = 900
# crop_top = image[upper: lower,:,:]
# cv2.imshow("Cropped Image", crop_top)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# his=cv2.calcHist([image],[0],None,[256],[0,256]) #none means using full image
# print(his)
# alpha=2
# beta=0
# new_image=cv2.convertScaleAbs(image,alpha=alpha,beta=beta)
# cv2.imshow("New Image", new_image)
# max_value=255
# thresh_value=80
# img=cv2.threshold(image,thresh_value,max_value,cv2.THRESH_BINARY)[1]
# cv2.imshow("Thresholded Image", img)
# resized=cv2.resize(image,None,fx=0.5,fy=1,interpolation=cv2.INTER_CUBIC)
# cv2.imshow("Resized Image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# kernel = np.ones((3,3), np.float32) / 9  
# # filtered = cv2.filter2D(image, -1, kernel)
# #filtered = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
# filtered = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_DEFAULT)
# cv2.imshow("Linear Filter", filtered)
# blur = cv2.GaussianBlur(image, (5,5), 0)
# cv2.imshow("Gaussian Blur", blur)
# kernel = np.array([
#     [0, -1, 0],
#     [-1, 5,-1],
#     [0, -1, 0]
# ])

# sharpened = cv2.filter2D(image, -1, kernel)
# cv2.imshow("Sharpened", sharpened)
# edges = cv2.Canny(image, 100, 200)
# cv2.imshow("Edges", edges)

# median = cv2.medianBlur(image, 5)
# cv2.imshow("Median Filter", median)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
# #Pixel > 127 → white and Pixel ≤ 127 → black
# cv2.imshow("Threshold", thresh)
