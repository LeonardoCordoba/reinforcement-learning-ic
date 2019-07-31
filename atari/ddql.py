import numpy as np
import os
import random
import shutil


GAMMA = 0.99
MEMORY_SIZE = 900000
BATCH_SIZE = 32
TRAINING_FREQUENCY = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 40000
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
REPLAY_START_SIZE = 50000

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_TEST = 0.02
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS

class DDQNNGame:

    def __init__(self, base_model, env, paths, ddqnn_params, train):
        self.base_model = base_model
        self.train = train # esto es para indicar si estamos entrenando o testeando
        if self.train:
            self.target_model = self.get_model_copy(self.base_model)
        self.env = env
        self.paths = paths
        self.ddqnn_params = ddqnn_params
        if self.train:
            self._reset_target_network()
        self.epsilon = self.ddqnn_params["exploration_max"]
        self.memory = []
        
    
    def get_model_copy(self, model):
        # TODO: ver como copiar modelo
        return model_copy
    
    def move(self, state):
        if self.train is False:
            if np.random.rand() < self.ddqnn_params["exploration_test"]:
                return random.randrange(self.env.action_space)
        else:
            if np.random.rand() < self.epsilon or len(self.memory) < self.ddqnn_params["replay_start_size"]:
                return random.randrange(self.env.action_space)
        
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

        if total_step % self.ddqnn_params["model_persistence_update_frequency"] == 0:
            self._save_model()

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
                q[entry["action"]] = entry["reward"] + GAMMA * next_q_value
            q_values.append(q)
            max_q_values.append(np.max(q))

        fit = self.base_model.fit(np.asarray(current_states).squeeze(),
                            np.asarray(q_values).squeeze(),
                            batch_size=self.ddqnn_params["batch_size"],
                            verbose=0)
        loss = fit.history["loss"][0]
        accuracy = fit.history["acc"][0]
        return loss, accuracy, np.mean(max_q_values)
    
    def _save_model(self):
        self.base_model.save_weights(self.paths["model"])