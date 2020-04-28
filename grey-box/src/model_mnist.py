import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, MaxPooling2D, ReLU)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model

from .callback import WeightsFreezer
from .metrics import F1Score


class SimpleNet(models.Sequential):
    def __init__(self):
        super().__init__()
        self.initialize_layers()
        loss_fn = SparseCategoricalCrossentropy()
        self.compile(
            optimizer="adam",
            loss="mse",
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(),
            ],
        )
        self.training_records = []

    def n_weights(self):
        return np.add.reduce(
            [np.multiply.reduce(l.shape) for l in self.trainable_weights]
        )

    def fit(self, *args, signature=None, **kwargs):
        if signature is not None:
            if signature.length > self.count_params():
                raise ValueError(
                    f"Signature length is {signature.length} but model has only {self.count_params()} parameters"
                )
            wf = WeightsFreezer(signature)
            kwargs.setdefault("callbacks", []).append(wf)
            print(f"Train model with signature of size {signature.length}")
        new_history = super().fit(*args, **kwargs)
        self.training_records.append(new_history)

    def plot_training(self):
        if not self.training_records:
            raise ValueError("Model has not been trained yet")
        else:
            records = self.merge_records()
            fig, ax = plt.subplots(1, 2)

            ax[0].plot(records["accuracy"], label="Train")
            ax[0].plot(records["val_accuracy"], label="Test")
            ax[0].set(xlabel="Epoch", ylabel="Accuracy", title="Model accuracy")
            ax[0].legend(loc="upper left")

            ax[1].plot(records["loss"], label="Train")
            ax[1].plot(records["val_loss"], label="Test")
            ax[1].set(xlabel="Epoch", ylabel="Loss", title="Model loss")
            ax[1].legend(loc="upper right")

            return fig, ax

    def merge_records(self):
        records = {}
        for record in self.training_records.copy():
            for metric, values in record.history.items():
                records.setdefault(metric, []).extend(values)
        return records

    def save_training_plot(self, fname):
        fig, ax = self.plot_training()
        fig.savefig(fname)

    def save_training_history(self, fname):
        pd.DataFrame(self.history.history).to_csv(fname)

    def initialize_layers(self):
        # Classifier block
        self.add(Dense(256, activation="relu", input_dim=256))
        self.add(Dense(128, activation="relu"))
        self.add(Dense(128, activation="relu"))
        self.add(Dense(128, activation="relu"))
        self.add(Dense(10, activation="linear"))
