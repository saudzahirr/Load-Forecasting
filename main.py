import re
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, SimpleRNN, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from datetime import datetime



class LoadForecasting:
    """
    A class for load forecasting using various deep learning models.

    Parameters:
    - folderpath (str): The folder path where the data file is located.
    - filename (str): The name of the data file.

    Methods:
    - load_data: Load data from a text file and prepare it for analysis.
    - prepare_dataset: Prepare the dataset by converting columns to dataframes and parsing dates.
    - df_to_X_y: Convert a dataframe to X and y for training machine learning models.
    - plot_graph: Plot a time series graph of a specific column.
    - plot_histogram: Plot a histogram for a specific column.
    - plot_kernel_distribution: Plot the probability density estimate (PDE) for a column.
    - plot_comparison: Plot a comparison between true values and predictions.
    - plot_predictions: Plot the predictions of a specific column against actual values.
    - rnn_lstm_gru: Train and evaluate an RNN-LSTM-GRU model for load forecasting.
    """
    def __init__(self, folderpath, filename):
        self.folderpath = folderpath
        self.filename = filename
        self.data = {}
        self.load_data()
        self.prepare_dataset()


    def load_data(self):
        filepath = os.path.join(self.folderpath, self.filename)
        with open(filepath, "r") as datafile:
            for line in datafile.readlines():
                line = line.strip()
                if re.match(r"[a-zA-Z]", line):
                    keys = line.split(";")
                    for key in keys:
                        self.data[key] = []
                else:
                    values = line.split(";")
                    if "?" not in values:
                        values[0] = values[0][:-2] + str(int(values[0][-2:]) + 12)
                        values[2:] = list(map(float, values[2:]))
                        for key, value in zip(keys, values):
                            self.data[key].append(value)


    def prepare_dataset(self):
        self.dataframes = {}
        date = self.data["Date"]
        time = self.data["Time"]

        for key in self.data:
            if key != "Date" and key != "Time":
                self.dataframes[key] = pd.DataFrame({
                    "Date": date,
                    "Time": time,
                    key: self.data[key]
                })

                self.dataframes[key]["Date-Time"] = pd.to_datetime(self.dataframes[key]["Date"] + ' ' + self.dataframes[key]["Time"], format='%d/%m/%Y %H:%M:%S')
                self.dataframes[key].index = self.dataframes[key]["Date-Time"]
                self.dataframes[key] = self.dataframes[key].drop(columns=["Date", "Time", "Date-Time"])


    @staticmethod
    def df_to_X_y(dataframe, window_size=5):
        """
        Convert a dataframe to X and y for training machine learning models.

        Args:
        - dataframe (pd.DataFrame): The input dataframe.
        - window_size (int): The size of the sliding window for feature extraction.

        Returns:
        - X (np.array): The input features.
        - y (np.array): The target values.
        """
        df_as_np = dataframe.to_numpy()
        X = []
        y = []
        for i in range(len(df_as_np) - window_size):
            row = [[a] for a in df_as_np[i:i + window_size]]
            X.append(row)
            label = df_as_np[i + window_size]
            y.append(label)
        return np.array(X), np.array(y)


    def plot_graph(self, column):
        x = self.dataframes[column].index[-100:]
        y = self.data[column][-100:]

        upper_bound = y + 0.5*np.sqrt(np.mean(y)/np.std(y))
        lower_bound = y - 0.5*np.sqrt(np.mean(y)/np.std(y))

        trace = go.Scatter(
            x=x,
            y=y,
            mode='lines',
            line=dict(color='blue'),
            name=f'{" ".join(column.split("_"))}'
        )

        trace_upper = go.Scatter(
            x=x,
            y=upper_bound,
            fill=None,
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            name='Upper Bound',
        )

        trace_lower = go.Scatter(
            x=x,
            y=lower_bound,
            fill='tonexty',
            fillcolor='rgba(0, 100, 80, 0.2)',
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            name='Lower Bound',
        )

        data = [trace, trace_upper, trace_lower]

        layout = go.Layout(
            title=f'{" ".join(column.split("_"))}',
            xaxis=dict(title='Date-Time'),
            yaxis=dict(title=f'{" ".join(column.split("_"))}'),
        )

        fig = go.Figure(data=data, layout=layout)
        fig.write_image(os.path.join(self.folderpath, f"{column}_plot.png"))
        fig.show()


    def plot_histogram(self, column):
        data = self.dataframes[column]
        filename = f"{column.lower()}.png"
        xlabel = column
        
        fig = px.histogram(data, nbins=30)
        fig.update_layout(title=f'{" ".join(column.split("_"))}', xaxis_title=xlabel, yaxis_title="Frequency")
        fig.write_image(os.path.join(self.folderpath, filename))
        fig.show()


    def plot_kernel_distribution(self, column):
        data = self.dataframes[column]
        filename = f"{column.lower()}_pde.png"
        xlabel = f"{column} Probability Density"

        fig = px.histogram(data, nbins=30, histnorm="probability density")
        fig.update_layout(title=f'{" ".join(column.split("_"))}', xaxis_title=xlabel, yaxis_title="Probability Density")
        fig.write_image(os.path.join(self.folderpath, filename))
        fig.show()


    def plot_comparison(self, y_test, y_pred, column, filename):
        y_test = y_test.flatten()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test[:1000], y=y_pred[:1000], mode="markers"))
        fig.update_layout(title=f'{" ".join(column.split("_"))}', xaxis_title="True Values", yaxis_title="Predictions")
        fig.write_image(os.path.join(self.folderpath, f"{filename}_test_comparison.png"))
        fig.show()


    def plot_predictions(self, time, y_test, y_pred, column, filename):
        y_test = y_test.flatten()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=y_pred[-100:], mode='lines', name='Predictions'))
        fig.add_trace(go.Scatter(x=time, y=y_test[-100:], mode='lines', name='Actual'))

        fig.update_layout(
            title=f'Predictions vs Actual {" ".join(column.split("_"))}',
            xaxis_title="Date-Time",
            yaxis_title=f'{" ".join(column.split("_"))}',
        )
        fig.write_image(os.path.join(self.folderpath, f"{filename}_test_predictions.png"))
        fig.show()


    def rnn_lstm_gru(self, column, WINDOW_SIZE=5):
        X, y = self.df_to_X_y(self.dataframes[column], WINDOW_SIZE)
        print(f"{column}: X.shape = {X.shape}, y.shape = {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Build an RNN-LSTM-GRU model
        model = Sequential()
        model.add(InputLayer((WINDOW_SIZE, 1)))
        model.add(SimpleRNN(32, return_sequences=True, activation=tf.nn.leaky_relu))
        model.add(LSTM(64, return_sequences=True, activation=tf.nn.leaky_relu))
        model.add(LSTM(64, return_sequences=True, activation=tf.nn.leaky_relu))
        model.add(GRU(32, activation=tf.nn.leaky_relu))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='linear'))
        
        model.summary()

        check_point = ModelCheckpoint(f'model/{column}_best_model.h5', save_best_only=True)
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])

        early_stopping = EarlyStopping(monitor="val_loss", patience=0.1)
        history = model.fit(X_train, y_train, validation_split=0.1, epochs=200, batch_size=2**12, callbacks=[check_point, early_stopping])

        test_predictions = model.predict(X_test).flatten()

        print(f"y_test = {y_test.flatten()}")
        print(f"y_pred = {test_predictions}")
        print("MSE = ", mse(y_test, test_predictions))

        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=list(range(1, len(history.history['loss']) + 1)), y=history.history['loss'], mode='lines', name='Training Loss'))
        loss_fig.add_trace(go.Scatter(x=list(range(1, len(history.history['val_loss']) + 1)), y=history.history['val_loss'], mode='lines', name='Validation Loss'))
        loss_fig.update_layout(title=f'{" ".join(column.split("_"))} Loss', xaxis_title="Epochs", yaxis_title="Loss")
        loss_fig.write_image(os.path.join(self.folderpath, f"{column.lower()}_loss.png"))
        loss_fig.show()

        self.plot_comparison(y_test, test_predictions, column, column.lower())
        self.plot_predictions(self.dataframes[column].index[-100:], y_test, test_predictions, column, column.lower())



def main():
    folderpath = "Load_Forecasting/"
    filename = "household_power_consumption.txt"

    lf = LoadForecasting(folderpath, filename)

    for column in lf.dataframes:
        lf.plot_graph(column)
        lf.plot_histogram(column)
        lf.plot_kernel_distribution(column)
        lf.rnn_lstm_gru(column, WINDOW_SIZE = 10)



if __name__ == "__main__":
    main()
