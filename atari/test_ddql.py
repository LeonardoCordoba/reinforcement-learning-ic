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
import numpy as np
from model import get_predefined_model
from ddql import DDQNNGame
from preprocessing import scale_color, wrap_deepmind
import gym
import random
import matplotlib.pyplot as plt
import os
import tensorflow as tf
#!pip install gym[atari]


#%%
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

# requirements:
# Open AI Gym (pip install gym[all])
# OpenCV
# JSAnimation - Only for Jupyter Display
# ImageIO 


#%%
# %% Instantiate model
# ["full", "1,5M", "800k", "300k", "100k"]
MODEL_NAME = "800k"
cnn = get_predefined_model(MODEL_NAME, INPUT_SHAPE)
cnn_2 = get_predefined_model(MODEL_NAME, INPUT_SHAPE)


#%%
# %% Setup
path = os.getcwd()

# %% gen path

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

# Para poder estudiar si se estaba entrenando tuve que cambiar dos parametros
# batch_size lo baje de 32 a 16 - vuelve a 32
# replay_start_size lo baje de 50000 a 500 - lo paso a 5000
# Ademas cambio y agregue muchos mas episodios y corridas

train = True

game_model = DDQNNGame(cnn, cnn_2, env, paths, params, train)

# Agregue los logs a TensorBoard pero no me funciono
#keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True)

env.reset()
frameshistory = []
done = False
model_save_freq = 10000 # 50000
total_step_limit = 1000000 # 10000000
total_run_limit = 1000 # 5000
render = False #True
clip = True

run = 0
total_step = 0
saves = 0

# TODO Hacer que el modelo cree una carpeta dentro de models para guardar las distintas versiones o que lo
# guarde en el nombre del archivo asi como guardamos las versiones de entrenamiento intermedias 
# TODO Ordenar el codigo
# TODO TOTHINK Hacer esto en un ejecutable .py o una funcion que guarde logs para análisis 
# posterior y tome como argumentos el modelo, cantidad de partidas, cantidad de steps, 
# tiempo de guardado y variables de exploracion 
# TODO TOTHINK Jugar un poco con las variables de exploracion
# TODO Terminar las funciones de evaluacion de modelos

#%%
# %% Main loop
import time
exit = 0
print("Partida número: ", run)
print(game_model._weigths_snapshot())
start = time.time()
while exit == 0:
    
    run += 1
    current_state = env.reset()
    if WRAPPER != "DM":
        current_state = np.reshape(current_state, (84, 84, 1))

    step = 0
    score = 0
    while exit == 0:
        if total_step >= total_step_limit:
            print ("Reached total step limit of: " + str(total_step_limit))
            # No sería mejor un break?
            print("Tiempo transcurrido de corrida {}".format(time.time()-start))
            exit = 1
        total_step += 1
        step += 1

        if render:
            env.render()

        if total_step % model_save_freq == 0:
            # Cada model_save_freq de pasos totales guardo los pesos del modelo
            game_model.save_model(saving_path + "/model{}_freq{}K_run{}M_games{}K_copy{}.h5".format(MODEL_NAME, model_save_freq/1000, total_step_limit/1000000,total_run_limit/1000, saves))
            saves += 1

        action = game_model.move(current_state)
        next_state, reward, terminal, info = env.step(action)
        if WRAPPER != "DM":
            next_state = np.reshape(next_state, (84, 84, 1))

        # next_state = scale_color(next_state)

        if clip:
            reward = np.sign(reward)
        score += reward
        
        game_model.remember(current_state, action, reward, next_state, terminal)
        current_state = next_state

        game_model.step_update(total_step)

        if terminal:
            # game_model.save_run(score, step, run)
            if run % 50 == 0:
                weights_snap = game_model._weigths_snapshot()
                print("Partida número: ", run)
                print("Pesos modelo base: ", weights_snap[0])
                print("Pesos modelo copia: ", weights_snap[1])
                print(score)
                print("Tiempo transcurrido de corrida {}".format(time.time()-start))
            game_model._save_model()
            break
           
    # Corto por episodios
    if total_run_limit is not None and run >= total_run_limit:
        print ("Reached total run limit of: " + str(total_run_limit))
        print("Tiempo transcurrido de corrida {}".format(time.time()-start))
        exit = 1
        
final = time.time()


#%%
#get_ipython().run_line_magic('load_ext', 'tensorboard')


#%%
#get_ipython().run_line_magic('tensorboard', '--logdir logs')


#%%
game_model.base_model.__dict__


#%%
game_model.base_model.layers[1]._trainable_weights[0]._snapshot.__dict__


#%%
# https://stackoverflow.com/questions/43715047/keras-2-x-get-weights-of-layer
import tensorflow as tf
from tensorflow.contrib.keras import layers

input_x = tf.placeholder(tf.float32, [None, 10], name='input_x')    
dense1 = layers.Dense(10, activation='relu')
y = dense1(input_x)

weights = dense1.get_weights()


#%%
# resp 2 https://stackoverflow.com/questions/43715047/keras-2-x-get-weights-of-layer
weigths = [] 
# lista de listas que contiene capa por capa y en cada capa contiene en el primer elemento los pesos y los bias
for layer in game_model.base_model.layers: 
    print(layer.get_config())
    weigths.append(layer.get_weights())
    weigths[4][0].sum()
    #print(layer.get_config(), layer.get_weights())


#%%
weigths[3]


#%%
weigths[4][0].sum()


