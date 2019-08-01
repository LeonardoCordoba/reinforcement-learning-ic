import gym
import random
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation

# Para Jupyter notebook
#conda install -c conda-forge jsanimation / pip install JSAnimation

def display_frames_as_gif_notebook(frames, filename_gif = None):
    """
    Displays a list of frames as a gif, with controls
    """
    from ipywidgets import widgets
    from IPython.display import display
    from JSAnimation.IPython_display import display_animation
    
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    if filename_gif: 
        anim.save(filename_gif, writer = 'imagemagick', fps=20)
    display(display_animation(anim, default_mode='loop'))

# Como visualizar una trayectoria en una jupyter notebook. Todo esto solo fue probado en notebook
#env.reset()
#frameshistory=[]
#done=False
#while not done:
#    action = random.sample(list(range(6)), k=1)
#    obs, reward, done, info = env.step(action)
#    frameshistory.append(obs)

#display_frames_as_gif_notebook(frameshistory, 'playing_space_invaders_random.gif')


# Plot a trajectory

import gym
import random
import matplotlib.pyplot as plt
#import cv2

env = gym.make('SpaceInvaders-v0')
env.reset()
frameshistory=[]
done=False
while not done:
      action = random.sample(list(range(6)), k=1)
      obs, reward, done, info = env.step(action)
      frameshistory.append(obs)

display_frames_as_gif(frameshistory, 'playing_space_invaders_random.gif')

import imageio
from PIL import Image

imageio.mimsave('atari/trajectories/playing_space_invaders_random.gif', frameshistory, fps=30)
#img = Image.open('atari/trajectories/playing_space_invaders_random.gif').convert('RGB')

def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(60):
    x += np.pi / 15.
    y += np.pi / 20.
    im = plt.imshow(f(x, y), animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

# ani.save('dynamic_images.mp4')

plt.show()

# otra opcion

import datetime
import imageio

VALID_EXTENSIONS = ('png', 'jpg')


def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    imageio.mimsave(output_file, images, duration=duration)