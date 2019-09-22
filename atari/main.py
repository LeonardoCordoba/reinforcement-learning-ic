#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'atari'))
	print(os.getcwd())
except:
	pass

#%%
from keras.optimizers import RMSprop
import keras
from keras.models import load_model
import numpy as np
from model import get_predefined_model
from ddql import DDQNNGame
from preprocessing import scale_color, wrap_deepmind
import gym
import random
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.models import model_from_json
import json

# %% Set env
WRAPPER = "DM" 
env = gym.make('SpaceInvaders-v0')
path = os.getcwd()

if WRAPPER == "DM":
    env = wrap_deepmind(env, frame_stack=True)
    INPUT_SHAPE = (84, 84, 4)
else:
    from gym.wrappers import AtariPreprocessing
    env = AtariPreprocessing(env)
    INPUT_SHAPE = (84, 84, 1)

# %% Instantiate model
# ["full", "1,5M", "800k", "300k", "100k"]
MODEL_NAME = "800k"

with open('model/json/' + MODEL_NAME + '.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights("model/800k_1/model800k.h5")

# %% Setup
path = os.getcwd()


train = False
paths = {}
params = {"exploration_test":0.005}
game_model = DDQNNGame(model, model, env, paths, params, train)

game_model.play(env=env, model_save_freq=0, save=False, saving_path="model/800k_1",
                  total_step_limit=1000000, total_run_limit=1000, render=False,
                  clip=True, wrapper=WRAPPER, model_name="800k")


#%%
