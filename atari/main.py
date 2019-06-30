from keras.optimizers import RMSprop
from model import CNNModel

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
