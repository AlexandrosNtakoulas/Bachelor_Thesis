import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List, Tuple

class MachineLearningModel:
    def __init__(self,
                 features:List[str] =None,
                 target:str ="disp_speed",
                 test_size:float =0.1,
                 n_epoch:int =250,
                 batch_size:int =100,
                 learning_rate:float =0.01,
                 neurons:int =16):
        """
        Initialize the ML model class.
        """
        self.features = features
        self.target = target
        self.test_size = test_size
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.neurons = neurons

        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        self.feature_names = features
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def prepare_data(self, df:pd.DataFrame =None):
        """Extract features and target, split into train/val/test, scale features."""
        y = df[self.target]
        X = df[self.features].values
        X = self.scaler.fit_transform(X)

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        # validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.test_size, random_state=42
        )

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        return self

    def build_model(self):
        """Construct and compile the neural network."""
        inp = tf.keras.Input(shape=(self.X_train.shape[-1],), name="input_layer")
        x = tf.keras.layers.Dense(self.neurons, activation="relu")(inp)
        x = tf.keras.layers.Dense(self.neurons, activation=tf.nn.leaky_relu)(x)
        x = tf.keras.layers.Dense(self.neurons, activation=tf.nn.leaky_relu)(x)
        out = tf.keras.layers.Dense(1)(x)

        self.model = tf.keras.Model(inputs=inp, outputs=out)
        optim = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optim, loss="mean_absolute_error")
        return self

    def train(self):
        """Train the neural network."""
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.n_epoch,
            batch_size=self.batch_size,
            verbose=1,
            validation_data=(self.X_val, self.y_val),
        )
        return self

    def plot_loss(self):
        """Plot training and validation loss curves."""
        training_loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        epoch_count = range(1, len(training_loss) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epoch_count, training_loss, label="Training Loss")
        ax.plot(epoch_count, val_loss, label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.legend()
        plt.title(
            f"Training size: {self.X_train.shape[0]}, Validation size: {self.X_val.shape[0]}"
        )
        plt.show()

    def evaluate(self):
        """Scatter plot of true vs predicted."""
        y_pred = self.model.predict(self.X_test)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(self.y_test, y_pred, alpha=0.3)
        ax.plot(self.y_test, self.y_test, color="red", label="y = x")
        ax.set_xlabel(f"{self.target} true")
        ax.set_ylabel(f"{self.target} predicted")
        ax.legend()
        plt.title(f"Testing size: {self.X_test.shape[0]}")
        plt.show()

    def shap_summary(self, max_points=500):
        """SHAP feature importance summary plot."""
        X_shap = self.X_val[:max_points]
        explainer = shap.Explainer(self.model, X_shap)
        shap_values = explainer(X_shap)
        shap.summary_plot(shap_values, X_shap, feature_names=self.feature_names, plot_type="layered_violin")