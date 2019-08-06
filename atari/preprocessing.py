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

def scale_color(image, dim = (84, 84), gray_scale = True):
    """
    INPUT: 
        - image es una observacion de atari, es decir un numpy ndarray
        - dim es una 2d tupla con la escala de salida
        - gray_scale es un bool , default es True
    OUTPUT:
        - numpy ndarray con el color y la escala especificada
    """
    assert isinstance(image, np.ndarray), 'La imagen no es del tipo correcto!'
    if gray_scale:
        return cv2.cvtColor(cv2.resize(image, dsize = dim), cv2.COLOR_RGB2GRAY)
    else:
        return cv2.resize(image, dsize = dim)

# Ejemplo de como ejecutarlo     
# plt.imshow(scale_color(img),cmap = plt.get_cmap('gray'))