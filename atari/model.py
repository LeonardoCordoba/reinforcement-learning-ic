from keras.optimizers import RMSprop, Adam
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense


class CNNModel:
    """
    Keras wrapper to build CNN to play Atari. Based on different papers.
    """

    def __init__(self):
        self.model = Sequential()

    def set_model_params(self, parameters):
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
        assert "cnn" in parameters.keys(), """parameters must have a cnn key
        with a list of Conv2D dicts hyperparameters"""

        assert "dense" in parameters.keys(), """parameters must have a dense key
        with a list of dicts Dense hyperparameters"""

        assert "compiler" in parameters.keys(), """parameters must have a
        compiler key with a dict with loss, optimizer and metrcs
        hyperparameters"""

        cnn_layers = parameters["cnn"]
        dense_layers = parameters["dense"]
        compiler = parameters["compiler"]

        # First, add CNN layers
        for cnn_hp in cnn_layers:
            self.model.add(Conv2D(filters=cnn_hp["filters"],
                                  kernel_size=cnn_hp["kernel_size"],
                                  strides=cnn_hp["strides"],
                                  padding=cnn_hp["padding"],
                                  activation=cnn_hp["activation"],
                                  input_shape=cnn_hp["input_shape"],
                                  data_format=cnn_hp["data_format"])
                           )

        # Second, flatten
        self.model.add(Flatten())

        # Third, add dense layers
        for dense_hp in dense_layers:
            if "activation" not in dense_hp.keys():
                activation = None
            else:
                activation = dense_hp["activation"]
            self.model.add(Dense(units=dense_hp["units"],
                                 activation=activation))

        # Last, compile
        self.model.compile(loss=compiler["loss"],
                           optimizer=compiler["optimizer"],
                           metrics=compiler["metrics"])

        return self.model.summary()

def get_predefined_model(model_name, input_shape):
    if model_name == "original":
        model = CNNModel()

        conv_1 = {"filters": 32, "kernel_size": 8, "strides": (4, 4), "padding": "valid",
                  "activation": "relu", "input_shape": input_shape,
                  "data_format":"channels_last"}

        conv_2 = {"filters": 64, "kernel_size": 4, "strides": (2, 2), "padding": "valid",
                  "activation": "relu", "input_shape": input_shape,
                  "data_format": "channels_last"}

        conv_3 = {"filters": 64, "kernel_size": 3, "strides": (1, 1), "padding": "valid",
                  "activation": "relu", "input_shape": input_shape,
                  "data_format": "channels_last"}

        dense_1 = {"units": 512, "activation": "relu"}

        dense_2 = {"units": 6}

        compiler = {"loss": "mean_squared_error", "optimizer": RMSprop(lr=0.00025,
                                                                       rho=0.95,
                                                                       epsilon=0.01),
                    "metrics": ["accuracy"]}

        hp = {"cnn": [conv_1, conv_2, conv_3], "dense": [dense_1, dense_2],
              "compiler": compiler}

        model.set_model_params(hp)
    elif model_name == "little_a":
        model = CNNModel()

        conv_1 = {"filters": 16, "kernel_size": 4, "strides": (2, 2), "padding": "valid",
                "activation": "relu", "input_shape": input_shape,
                "data_format":"channels_last"}

        conv_2 = {"filters": 32, "kernel_size": 3, "strides": (1, 1), "padding": "valid",
                "activation": "relu", "input_shape": input_shape,
                "data_format": "channels_last"}

        dense_1 = {"units": 128, "activation": "relu"}

        dense_2 = {"units": 6}

        compiler = {"loss": "mean_squared_error", "optimizer": RMSprop(lr=0.00025,
                                                                    rho=0.95,
                                                                    epsilon=0.01),
                    "metrics": ["accuracy"]}

        hp = {"cnn": [conv_1, conv_2], "dense": [dense_1, dense_2],
            "compiler": compiler}

        model.set_model_params(hp)
    
    return model