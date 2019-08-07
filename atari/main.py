from keras.optimizers import RMSprop
import numpy as np
from atari.model import CNNModel
from atari.ddql import DDQNNGame

# REQUIREMENTS:
# Open AI Gym (pip install gym[all])
# OpenCV
# JSAnimation - Only for Jupyter Display
# ImageIO 

# Instantiate model
cnn = CNNModel()

conv_1 = {"filters": 32, "kernel_size": 8, "strides": (4, 4), "padding": "valid",
          "activation": "relu", "input_shape": (4, 84, 84),
          "data_format":"channels_first"}

conv_2 = {"filters": 64, "kernel_size": 4, "strides": (2, 2), "padding": "valid",
          "activation": "relu", "input_shape": (4, 84, 84),
          "data_format": "channels_first"}

conv_3 = {"filters": 64, "kernel_size": 3, "strides": (1, 1), "padding": "valid",
          "activation": "relu", "input_shape": (4, 84, 84),
          "data_format": "channels_first"}

dense_1 = {"units": 512, "activation": "relu"}

dense_2 = {"units": 6}

compiler = {"loss": "mean_squared_error", "optimizer": RMSprop(lr=0.00025,
                                                               rho=0.95,
                                                               epsilon=0.01),
            "metrics": ["accuracy"]}

hp = {"cnn": [conv_1, conv_2, conv_3], "dense": [dense_1, dense_2],
      "compiler": compiler}


cnn.set_model_params(hp)


# Play random
import gym
import random
import matplotlib.pyplot as plt
#import cv2

env = gym.make('SpaceInvaders-v0')
paths = {"model":"model/model.h5"}

exploration_max = 1.0
exploration_min = 0.1
exploration_steps = 850000
exploration_decay = (exploration_max-exploration_min)/exploration_steps


params = {"gamma":0.99, "memory_size": 900000, "batch_size": 32,
            "training_frequency": 4, "target_network_update_frequency": 40000,
            "model_persistence_update_frequency": 10000,
            "replay_start_size": 50000 ,"exploration_test": 0.02,
            "exploration_max": exploration_max, 
            "exploration_min": exploration_min,
            "exploration_steps": exploration_steps, 
            "exploration_decay": exploration_decay}
train = True

game_model = DDQNNGame(cnn, env, paths, params, train)

env.reset()
frameshistory=[]
done=False
total_step_limit = 10000
total_run_limit = 10
render = True
clip = True

run = 0
total_step = 0

# GAS!! Si corres lo que sigue parece que va bien hasta que da el siguiente error:
# ValueError: Error when checking input: expected conv2d_4_input to have shape (4, 84, 84) but got array with shape (210, 160, 3) 

# Entiendo que es porque lo que se trata de pasar es un frame a color, en rgb, por eso el 3 en (210, 160, 3)
# Supongo que habria que preprocesar los frames para resolverlo
# El error esta en la linea 101 de ddql --> 101             next_state_prediction = self.target_model.predict(next_state).ravel()

while True:
      if total_run_limit is not None and run >= total_run_limit:
            # print "Reached total run limit of: " + str(total_run_limit)
            exit(0)

      run += 1
      current_state = env.reset()
      # TODO: capaz se puede preprocesar aca currect_state y concatenar 4
      step = 0
      score = 0
      while True:
            if total_step >= total_step_limit:
                  # print "Reached total step limit of: " + str(total_step_limit)
                  exit(0)
            total_step += 1
            step += 1

            if render:
                  env.render()

            action = game_model.move(current_state)
            next_state, reward, terminal, info = env.step(action)
            if clip:
                  reward = np.sign(reward)
            score += reward
            game_model.remember(current_state, action, reward, next_state, terminal)
            current_state = next_state

            game_model.step_update(total_step)

            if terminal:
                  # game_model.save_run(score, step, run)
                  print(score)
                  break