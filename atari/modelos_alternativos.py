from atari.model import CNNModel, get_predefined_model
from keras.optimizers import RMSprop, Adam

input_shape = (84, 84, 4)
# Modelos actuales
get_predefined_model("original", input_shape)
# Trainable params: 1,684,128
get_predefined_model("little_a", input_shape)
# Trainable params: 1,484,464

# 1) Modelo muy chiquito: 337,584
model = CNNModel()

conv_1 = {"type": "cnn",
        "filters": 16, "kernel_size": 4, "strides": (2, 2), "padding": "valid",
        "activation": "relu", "input_shape": input_shape,
        "data_format":"channels_last"}

maxpool_1 = {"type": "maxpool",
        "pool_size": (2,2), "strides": None, "padding": "valid",
        "data_format": "channels_last"}

conv_2 = {"type": "cnn",
        "filters": 32, "kernel_size": 3, "strides": (1, 1), "padding": "valid",
        "activation": "relu", "input_shape": input_shape,
        "data_format": "channels_last"}

maxpool_2 = {"type": "maxpool",
        "pool_size": (2,2), "strides": None, "padding": "valid",
        "data_format": "channels_last"}

dense_1 = {"type": "dense", "units": 128, "activation": "relu"}

dense_2 = {"type": "dense","units": 6}

compiler = {"loss": "mean_squared_error", "optimizer": RMSprop(lr=0.00025,
                                                            rho=0.95,
                                                            epsilon=0.01),
            "metrics": ["accuracy"]}

flatten = {"type": "flatten"}
hp = {"layers": [conv_1, maxpool_1, conv_2, maxpool_2, flatten, dense_1, dense_2], 
        "compiler": compiler}
model.set_model_params(hp, input_shape)

# Modelo aún más chiquito: 98,032
model = CNNModel()

conv_1 = {"type": "cnn",
        "filters": 16, "kernel_size": 4, "strides": (2, 2), "padding": "valid",
        "activation": "relu", "input_shape": input_shape,
        "data_format":"channels_last"}

maxpool_1 = {"type": "maxpool",
        "pool_size": (2,2), "strides": None, "padding": "valid",
        "data_format": "channels_last"}

conv_2 = {"type": "cnn",
        "filters": 32, "kernel_size": 3, "strides": (1, 1), "padding": "valid",
        "activation": "relu", "input_shape": input_shape,
        "data_format": "channels_last"}

maxpool_2 = {"type": "maxpool",
        "pool_size": (2,2), "strides": None, "padding": "valid",
        "data_format": "channels_last"}

conv_3 = {"type": "cnn",
        "filters": 64, "kernel_size": 3, "strides": (1, 1), "padding": "valid",
        "activation": "relu", "input_shape": input_shape,
        "data_format": "channels_last"}

maxpool_3 = {"type": "maxpool",
        "pool_size": (2,2), "strides": None, "padding": "valid",
        "data_format": "channels_last"}

dense_1 = {"type": "dense", "units": 128, "activation": "relu"}

dense_2 = {"type": "dense","units": 6}

compiler = {"loss": "mean_squared_error", "optimizer": RMSprop(lr=0.00025,
                                                            rho=0.95,
                                                            epsilon=0.01),
            "metrics": ["accuracy"]}

flatten = {"type": "flatten"}
hp = {"layers": [conv_1, maxpool_1, conv_2, maxpool_2, conv_3, maxpool_3, flatten, dense_1, dense_2], 
        "compiler": compiler}
model.set_model_params(hp, input_shape)

# 3) Modelo intermedio A: 1,332,912
model = CNNModel()

conv_1 = {"type": "cnn",
        "filters": 16, "kernel_size": 4, "strides": (2, 2), "padding": "valid",
        "activation": "relu", "input_shape": input_shape,
        "data_format":"channels_last"}

maxpool_1 = {"type": "maxpool",
        "pool_size": (2,2), "strides": None, "padding": "valid",
        "data_format": "channels_last"}

conv_2 = {"type": "cnn",
        "filters": 32, "kernel_size": 3, "strides": (1, 1), "padding": "valid",
        "activation": "relu", "input_shape": input_shape,
        "data_format": "channels_last"}

dense_1 = {"type": "dense", "units": 128, "activation": "relu"}

dense_2 = {"type": "dense","units": 6}

compiler = {"loss": "mean_squared_error", "optimizer": RMSprop(lr=0.00025,
                                                            rho=0.95,
                                                            epsilon=0.01),
            "metrics": ["accuracy"]}

flatten = {"type": "flatten"}
hp = {"layers": [conv_1, maxpool_1, conv_2, flatten, dense_1, dense_2], 
        "compiler": compiler}
model.set_model_params(hp, input_shape)

# 4) Modelo intermedio B: 820,368
model = CNNModel()

conv_1 = {"type": "cnn",
        "filters": 16, "kernel_size": 4, "strides": (2, 2), "padding": "valid",
        "activation": "relu", "input_shape": input_shape,
        "data_format":"channels_last"}

maxpool_1 = {"type": "maxpool",
        "pool_size": (2,2), "strides": None, "padding": "valid",
        "data_format": "channels_last"}

dense_1 = {"type": "dense", "units": 128, "activation": "relu"}

dense_2 = {"type": "dense","units": 6}

compiler = {"loss": "mean_squared_error", "optimizer": RMSprop(lr=0.00025,
                                                            rho=0.95,
                                                            epsilon=0.01),
            "metrics": ["accuracy"]}

flatten = {"type": "flatten"}
hp = {"layers": [conv_1, maxpool_1, flatten, dense_1, dense_2], 
        "compiler": compiler}
model.set_model_params(hp, input_shape)
