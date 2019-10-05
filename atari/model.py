from keras.optimizers import RMSprop, Adam
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D


class CNNModel:
    """
    Keras wrapper to build CNN to play Atari. Based on different papers.
    """

    def __init__(self):
        self.model = Sequential()

    def set_model_params(self, parameters, input_shape):
        """Method used to pass parameters to CNNModel and build the NN.

        Parameters
        ----------
        parameters : dict
            Dict must have cnn and dense keys. Each key must have a list of
            dicts. Each dict in the list represents the parameters to build a
            layer in the model.
            For example:



        Returns
        -------
        type
            Description of returned object.

        """
        compiler = parameters["compiler"]

        # First, add CNN layers
        print(len(parameters["layers"]))
        for layer in parameters["layers"]:
            if layer["type"] == "cnn":
                print("added conv")
                self.model.add(Conv2D(filters=layer["filters"],
                                  kernel_size=layer["kernel_size"],
                                  strides=layer["strides"],
                                  padding=layer["padding"],
                                  activation=layer["activation"],
                                  input_shape=layer["input_shape"],
                                  data_format=layer["data_format"])
                                    )
                                    
            elif layer["type"] == "maxpool":
                print("added maxpool")
                self.model.add(MaxPooling2D(pool_size=layer["pool_size"],
                                    strides=layer["strides"],
                                    padding=layer["padding"],
                                    data_format=layer["data_format"])
                                    )
            
            elif layer["type"] == "flatten":
                print("added flatten")
                self.model.add(Flatten())

            elif layer["type"] == "dense":
                print("added dense")
                if "activation" not in layer.keys():
                    self.model.add(Dense(units=layer["units"]))
                else:
                    activation = layer["activation"]
                    self.model.add(Dense(units=layer["units"],
                                 activation=activation))
        self.model.compile(loss=compiler["loss"],
                           optimizer=compiler["optimizer"],
                           metrics=compiler["metrics"])
        self.model.build(input_shape)
        return self.model.summary()

def get_predefined_model(model_name, input_shape):

    model = CNNModel()

    if model_name == "full":
        conv_1 = {"type": "cnn",
                  "filters": 32, "kernel_size": 8, "strides": (4, 4), "padding": "valid",
                  "activation": "relu", "input_shape": input_shape,
                  "data_format":"channels_first"}

        conv_2 = {"type": "cnn",
                  "filters": 64, "kernel_size": 4, "strides": (2, 2), "padding": "valid",
                  "activation": "relu", "input_shape": input_shape,
                  "data_format": "channels_first"}

        conv_3 = {"type": "cnn",
                  "filters": 64, "kernel_size": 3, "strides": (1, 1), "padding": "valid",
                  "activation": "relu", "input_shape": input_shape,
                  "data_format": "channels_first"}

        dense_1 = {"type":"dense",
                 "units": 512, "activation": "relu"}

        dense_2 = {"type":"dense",
                 "units": 6}

        compiler = {"loss": "mean_squared_error", "optimizer": RMSprop(lr=0.00025,
                                                                       rho=0.95,
                                                                       epsilon=0.01),
                    "metrics": ["accuracy"]}

        flatten = {"type": "flatten"}
        hp = {"layers": [conv_1,conv_2,conv_3,flatten,dense_1,dense_2],
                "compiler": compiler} 

    elif model_name == "1,5M":
        # TODO: need to check using new env
        # Intermedio: 1.484.464
        conv_1 = {"type": "cnn",
                "filters": 16, "kernel_size": 4, "strides": (2, 2), "padding": "valid",
                "activation": "relu", "input_shape": input_shape,
                "data_format":"channels_last"}

        conv_2 = {"type": "cnn",
                "filters": 32, "kernel_size": 3, "strides": (1, 1), "padding": "valid",
                "activation": "relu", "input_shape": input_shape,
                "data_format": "channels_last"}

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

        hp = {"layers": [conv_1, conv_2, maxpool_1, flatten, dense_1, dense_2], 
                "compiler": compiler}

    elif model_name == "300k":
        # TODO: need to check using new env
        #1) Modelo muy chiquito: 337,584
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

    elif model_name == "100k":
        # TODO: need to check using new env
        # Modelo aún más chiquito: 98,032
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

    elif model_name == "800k":
        # TODO: need to check using new env
        # Modelo intermedio B: 820,368
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
    return model