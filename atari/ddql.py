import numpy as np
import os
import random
import shutil
import keras
import time
import pandas as pd
# GAMMA = 0.99
# MEMORY_SIZE = 900000
# BATCH_SIZE = 32
# TRAINING_FREQUENCY = 4
# TARGET_NETWORK_UPDATE_FREQUENCY = 40000
# MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
# REPLAY_START_SIZE = 50000

# EXPLORATION_MAX = 1.0
# EXPLORATION_MIN = 0.1
# EXPLORATION_TEST = 0.02
# EXPLORATION_STEPS = 850000
# EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS

class DDQNNGame:

    def __init__(self, base_model, copy_model, env, paths, ddqnn_params, train):
        self.base_model = base_model.model
        self.train = train
        if self.train:
            self.target_model = copy_model.model
        self.env = env
        self.paths = paths
        self.ddqnn_params = ddqnn_params
        if self.train:
            self._reset_target_network()
            self.epsilon = self.ddqnn_params["exploration_max"]
        else:
            assert "exploration_test" in ddqnn_params.keys()
        self.memory = []
        
    
    def get_model_copy(self, model):
        # GAAAASSSS
        # TODO: chequear que funcione bien
        model_copy= keras.models.clone_model(model)
        # TODO: replace with number of variables in input layer
        # model_copy.build((None, 10)) 
        model_copy.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model_copy.set_weights(model.get_weights())
        return model_copy
    
    def move(self, state):
        if self.train is False:
            if np.random.rand() < self.ddqnn_params["exploration_test"]:
                return random.randrange(self.env.action_space.n)
        else:
            if np.random.rand() < self.epsilon or len(self.memory) < self.ddqnn_params["replay_start_size"]:
                return random.randrange(self.env.action_space.n)
        
        q_values = self.base_model.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        return np.argmax(q_values[0])
    
    def _reset_target_network(self):
        self.target_model.set_weights(self.base_model.get_weights())
    
    def remember(self, current_state, action, reward, next_state, terminal):
        self.memory.append({"current_state": current_state,
                        "action": action,
                        "reward": reward,
                        "next_state": next_state,
                        "terminal": terminal})
        if len(self.memory) > self.ddqnn_params["memory_size"]:
            self.memory.pop(0)

    def step_update(self, total_step):
        if len(self.memory) < self.ddqnn_params["replay_start_size"]:
            return None

        if total_step % self.ddqnn_params["training_frequency"] == 0:
            loss, accuracy, average_max_q = self._train()

        self._update_epsilon()

        # if total_step % self.ddqnn_params["model_persistence_update_frequency"] == 0:
        #     self._save_model()

        if total_step % self.ddqnn_params["target_network_update_frequency"] == 0:
            self._reset_target_network()
            print('{{"metric": "epsilon", "value": {}}}'.format(self.epsilon))
            print('{{"metric": "total_step", "value": {}}}'.format(total_step))

    def _update_epsilon(self):
        self.epsilon -= self.ddqnn_params["exploration_decay"]
        self.epsilon = max(self.ddqnn_params["exploration_min"], self.epsilon)

    def _train(self):
        batch = np.asarray(random.sample(self.memory, self.ddqnn_params["batch_size"]))
        if len(batch) < self.ddqnn_params["batch_size"]:
            return None

        current_states = []
        q_values = []
        max_q_values = []

        for entry in batch:
            current_state = np.expand_dims(np.asarray(entry["current_state"]).astype(np.float64), axis=0)
            current_states.append(current_state)
            next_state = np.expand_dims(np.asarray(entry["next_state"]).astype(np.float64), axis=0)
            next_state_prediction = self.target_model.predict(next_state).ravel()
            next_q_value = np.max(next_state_prediction)
            q = list(self.base_model.predict(current_state)[0])
            if entry["terminal"]:
                q[entry["action"]] = entry["reward"]
            else:
                q[entry["action"]] = entry["reward"] + self.ddqnn_params["gamma"] * next_q_value
            q_values.append(q)
            max_q_values.append(np.max(q))

        fit = self.base_model.fit(np.asarray(current_states).squeeze(),
                            np.asarray(q_values).squeeze(),
                            batch_size=self.ddqnn_params["batch_size"],
                            verbose=0)
        loss = fit.history["loss"][0]
        accuracy = fit.history["acc"][0]
        return loss, accuracy, np.mean(max_q_values)

    def save_model(self, path):
        self.base_model.save_weights(path)

    def _weigths_snapshot(self):
        weigths_base = []
        weigths_target = []
        for layer_base, layer_target in zip(self.base_model.layers, self.target_model.layers):
            try:
                weigths_base.append(layer_base.get_weights()[0].sum())
                weigths_target.append(layer_target.get_weights()[0].sum())
            except IndexError:
                pass
        return weigths_base, weigths_target
    
    def play(self, env, save, saving_path, model_save_freq, total_step_limit,
                total_run_limit, render, clip, wrapper, model_name):

        exit = 0
        env.reset()
        frameshistory = []
        done = False
        run = 0
        total_step = 0
        saves = 0
        start = time.time()
        performance = []

        while exit == 0:
            run += 1
            current_state = env.reset()
            if wrapper != "DM":
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
                    
                if save and (total_step % model_save_freq == 0):
                    # Cada model_save_freq de pasos totales guardo los pesos del modelo
                    model_save_freq_k = int(model_save_freq/1000)
                    total_step_limit_m = int(total_step_limit/1000000)
                    total_run_limit_k = int(total_run_limit/1000)

                    full_path = saving_path + "/model" + model_name + \
                    "_freq" + str(model_save_freq_k) + "K_run" + \
                    str(total_step_limit_m) + "M_games" + \
                        str(total_run_limit_k) + "K_copy" + str(saves) + ".h5"
                    self.save_model(full_path)
                    saves += 1

                action = self.move(current_state)
                next_state, reward, terminal, info = env.step(action)
                if wrapper != "DM":
                    next_state = np.reshape(next_state, (84, 84, 1))

                # next_state = scale_color(next_state)

                if clip:
                    reward = np.sign(reward)
                score += reward
                
                if self.train:
                    self.remember(current_state, action, reward, next_state, terminal)

                current_state = next_state

                if self.train:
                    self.step_update(total_step)

                    if terminal:
                        performance.append({"run":run,
                                            "step":step,
                                            "score":score})
                        pd.DataFrame(performance).to_csv(saving_path + "/performance.csv", index=False)

                        # game_model.save_run(score, step, run)
                        if run % 50 == 0:
                            weights_snap = self._weigths_snapshot()
                            # print("Partida número: ", run)
                            # print("Pesos modelo base: ", weights_snap[0])
                            # print("Pesos modelo copia: ", weights_snap[1])
                            # print(score)
                            # print("Tiempo transcurrido de corrida {}".format(time.time()-start))
                        if save:
                            full_path = f"{saving_path}/model{model_name}.h5"
                            self.save_model(full_path)
                        break
                else:
                    if terminal:
                        performance.append({"run":run,
                                            "total_step":total_step,
                                            "score":score})
                        pd.DataFrame(performance).to_csv(saving_path + "/performance.csv", index=False)
                        break

            # Corto por episodios
            if total_run_limit is not None and run >= total_run_limit:
                # print ("Reached total run limit of: " + str(total_run_limit))
                # print("Tiempo transcurrido de corrida {}".format(time.time()-start))
                exit = 1
                
        final = time.time()
