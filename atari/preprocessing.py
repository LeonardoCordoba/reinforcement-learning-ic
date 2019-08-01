import gym
import cv2
import numpy as np 
import matplotlib.pyplot as plt

def rgb(array):
    # Ejemplo de como ejecutarlo 
    # img_gray = rgb(img)
    # plt.imshow(img_gray, cmap = plt.get_cmap('gray'), vmin = 0, vmax = 1)

    r,g,b = array[:,:,0], array[:,:,1], array[:,:,2]
    gray = 0.2989*r + 0.587*g + 0.114*b
    return gray

def image_to_gray(image, dim = (84, 84), gray_scale = True):
    """
    INPUT: 
        - image is an observation from an atari game from GYM so it is a numpy.ndarray
        - dim is a 2d tuple of the rescale
        - gray_scale is a boolean , default is true
    OUTPUT:
        -  
    """
    #assert image.istype(np.ndarray)
    if gray_scale:
        return cv2.cvtColor(cv2.resize(image, dsize = dim), cv2.COLOR_RGB2GRAY)
    else:
        return cv2.resize(image, dsize = dim)

# Ejemplo de como ejecutarlo     
# plt.imshow(preprocessing(img),cmap = plt.get_cmap('gray'))