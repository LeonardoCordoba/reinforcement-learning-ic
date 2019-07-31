import gym
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
# Para Jupyter notebook
#from ipywidgets import widgets
#from IPython.display import display
# pip install JSAnimation
#from JSAnimation.IPython_display import display_animation

def display_frames_as_gif(frames, filename_gif = None):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    if filename_gif: 
        anim.save(filename_gif, writer = 'imagemagick', fps=20)
    display(display_animation(anim, default_mode='loop'))

# Como visualizar una trayectoria. Todo esto solo fue probado en notebook
env.reset()
frameshistory=[]
done=False
while not done:
    action = random.sample(list(range(6)), k=1)
    obs, reward, done, info = env.step(action)
    frameshistory.append(obs)

display_frames_as_gif(frameshistory, 'playing_space_invaders_random.gif')