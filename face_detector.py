import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt
from os import listdir


def detect(file):
    #reading the image
    img=Image.open(file)
    img=img.convert('RGB')

    #converting image to numpy array
    img_array=np.asarray(img)


    face_detector=MTCNN()

    #detecting face using MTCNN object
    face_pixels=face_detector.detect_faces(img_array)

    #getting location of faces,eyes in the image
    for i in face_pixels:
        x1,y1,width,height=i['box']
        left_eye=i['keypoints']['left_eye']
        right_eye=i['keypoints']['right_eye']

        #drawing rectangle around the face
        cv2.rectangle(img_array,(x1,y1),(x1+width,y1+height),(0,255,0),2)

        #drawing circle around eyes of each face 
        cv2.circle(img_array,left_eye,5,(255,0,0),2)
        cv2.circle(img_array,right_eye,5,(255,0,0),2)

    #displaying the image with face and eyes ditected
    plt.imshow(img_array)
    plt.show()  


#location of the image folder
folder='/home/clown/test_images/'

for image in listdir(folder):
    path=folder+image
    detect(path)
