import matplotlib
import gym
import numpy as np
from keras.models import model_from_json, load_model
import pandas as pd


def estimate_perf(model_w, model_architecture = None, Nrep = 50):
    # TODO: Terminar - Pensar cual puede ser la mejor performance para devolver
    """
    INPUT: Model es un DDQNNGame, y N un entero que es la cantidad de partidas a hacerlo
            jugar para estimar la performance del modelo.
            https://jovianlin.io/saving-loading-keras-models/
            OJO: Si solo se guardan los pesos se necesita tambien tener la 
            arquitectura(.json) del modelo tambien.
    OUTPUT: Una tupla con la performance del modelo media y el desvio
    """

    perf_media = 0
    perf_std = 0
    # evaluar performance --> porcentaje de partidas ganadas(entonces no tengo un desvio)

    return perf_media, perf_std

def evaluate_model(model_list, steps_list, Nrep):
    # TODO: Devolver la performance como un dataframe
    # TODO: Dibujar performance segun steps de entrenamiento
    """
    INPUT: Model_list es una lista de DDQNNGames en los steps de entrenamiento(lista) y Nrep la
            cantidad de partidas que se usaran para evaluar los modelos
    """

    # asumimos que los modelos siempre fueron guardados con model_save(asi no necesitamos el .json)
    models = []
    for m in model_list:
        models.append(load_model(m))
    
    M = len(models)
    performance = dict(list(zip([steps for steps in steps_list], [0 for i in range(M)])))
    for model, steps in zip(models, steps_list):
        performance[steps] = estimate_perf(model, Nrep)

    return None

def compare_models():
    # TODO: Empezar, dadas varias arquitecturas con settings diferentes, compararlos

    return None