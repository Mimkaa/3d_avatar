import cv2
import numpy as np
import pygame as pg
from PIL import Image
from settings import *
def pg_pil_opencv(image):
    raw_str = pg.image.tostring(image, "RGB", False)
    img = Image.frombytes("RGB", image.get_size(), raw_str)
    open_cv_image = np.array(img)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def cv2ImageToSurface(cv2Image):
    if cv2Image.dtype.name == 'uint16':
        cv2Image = (cv2Image / 256).astype('uint8')
    size = cv2Image.shape[1::-1]
    if len(cv2Image.shape) == 2:
        cv2Image = np.repeat(cv2Image.reshape(size[1], size[0], 1), 3, axis = 2)
        format = 'RGB'
    else:
        format = 'RGBA' if cv2Image.shape[2] == 4 else 'RGB'
        cv2Image[:, :, [0, 2]] = cv2Image[:, :, [2, 0]]
    surface = pg.image.frombuffer(cv2Image.flatten(), size, format)
    return surface.convert_alpha() if format == 'RGBA' else surface.convert()

def opencv_pil_pg(img):
    color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_converted)
    pg_image=pg.image.fromstring(pil_image.tobytes(), pil_image.size, "RGB").convert_alpha()
    pg_image.set_colorkey(BLACK)
    return pg_image

def distortion(im,size,old_coords,new_coords):
    img=pg_pil_opencv(im)
    pts1=np.float32(old_coords)
    #new_coords=[(int(i),int(j))for i,j in new_coords]
    pts2=np.float32(new_coords)
    #cv2.circle(img,(new_coords[-1][0],new_coords[-1][1]),3,(0,0,255))
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    output=cv2.warpPerspective(img,matrix,(size[0],size[1]))


    # cv2.imshow("out",output)
    #
    # cv2.waitKey(0)
    return opencv_pil_pg(output)

