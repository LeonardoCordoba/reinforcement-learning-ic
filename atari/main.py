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
MODEL_NAME = "100k"

cnn = get_predefined_model(MODEL_NAME, INPUT_SHAPE)
cnn_2 = get_predefined_model(MODEL_NAME, INPUT_SHAPE)

# with open('model/json/' + MODEL_NAME + '.json','r') as f:
#     model_json = json.load(f)

# model = model_from_json(model_json)
# model.load_weights("model/800k_1/model800k.h5")

# GasManija: mover a utils y armar lindo
def gen_path(path, model_name, exp_num):
    return os.path.join(path, "model", MODEL_NAME + "_" + str(exp_num))

exp_num = 1

while os.path.exists(gen_path(path, MODEL_NAME, exp_num)):
    exp_num += 1

saving_path = gen_path(path, MODEL_NAME, exp_num)
os.mkdir(saving_path)

paths = {"model":saving_path + "/model{}.h5".format(MODEL_NAME)}


#%% 
exploration_max = 1.0
exploration_min = 0.1
exploration_steps = 80000 # 800000
exploration_decay = (exploration_max-exploration_min)/exploration_steps

params = {"gamma":0.99, "memory_size": 900000, "batch_size": 32,
            "training_frequency": 4, "target_network_update_frequency": 40000,
            "model_persistence_update_frequency": 10000,
            "replay_start_size": 5000 ,"exploration_test": 0.02,
            "exploration_max": exploration_max, 
            "exploration_min": exploration_min,
            "exploration_steps": exploration_steps, 
            "exploration_decay": exploration_decay}


# %% Setup

train = True
paths = {}
# params = {"exploration_test":0.005}
game_model = DDQNNGame(cnn, cnn_2, env, paths, params, train)
game_model.play(env=env, model_save_freq=10000, save=True, saving_path="model/100k_1/",
                  total_step_limit=1000000, total_run_limit=5000, render=False,
                  clip=True, wrapper=WRAPPER, model_name="100k")



#%%
