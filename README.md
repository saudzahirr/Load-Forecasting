# Load-Forecasting

Imagine you're tasked with predicting future energy consumption, critical for effective energy management. This code offers a comprehensive solution for load forecasting using various deep learning models. It simplifies the process of data preparation, visualization, and model training, enabling you to make accurate load forecasts.

## Features

- **Data Loading and Preparation:** The code can load data from a text file, parse it, and create structured datasets for analysis. It handles date and time parsing, column conversion, and more.

- **Data Visualization:** It provides various methods for visualizing time series data, including plotting time series graphs, histograms, and probability density estimates.

- **Deep Learning Models:** The code includes an RNN-LSTM-GRU model for load forecasting. It sets up the model architecture, trains it on your data, and evaluates its performance.

- **Customization:** You can adjust parameters like window size, learning rate, and model architecture to suit your specific load forecasting needs.

## Usage

1. **Data Preparation:**
   - Ensure you have a data file named ["household_power_consumption.txt"](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) in the "Load_Forecasting" folder.
   - Make sure your data file follows the expected format for successful parsing.

2. **Visualization:**
   - The code generates time series graphs, histograms, and probability density estimate plots for each column in your dataset. These visualizations help you understand the data's characteristics.

3. **Model Training:**
   - The code trains an RNN-LSTM-GRU model for load forecasting. It preprocesses the data, splits it into training and testing sets, and trains the model.
   - You can customize the model architecture, learning rate, and other parameters in the `rnn_lstm_gru` method.

4. **Evaluation:**
   - The code evaluates the model's performance using Mean Squared Error (MSE) and provides visualizations to compare predicted values with actual values.

## Methods

### LoadForecasting Class
- `load_data`: Load data from a text file and prepare it for analysis.
- `prepare_dataset`: Prepare the dataset by converting columns to dataframes and parsing dates.
- `df_to_X_y`: Convert a dataframe to X and y for training machine learning models.
- `plot_graph`, `plot_histogram`, `plot_kernel_distribution`: Plot data visualizations.
- `plot_comparison`, `plot_predictions`: Visualize model performance.
- `rnn_lstm_gru`: Train and evaluate an RNN-LSTM-GRU model for load forecasting.

### Main Function
- The `main` function initializes an instance of the `LoadForecasting` class and calls various methods to perform data visualization and load forecasting for each column in the dataset.

## Requirements

- Python 3.7 or later
- Libraries: `numpy`, `pandas`, `plotly`, `scikit-learn`, `tensorflow`, `datetime`

## Customization

You can customize various aspects of the code to fit your specific load forecasting project, including:
- Data file name and location
- Model architecture and hyperparameters
- Plotting options and visualizations

## Example

Here's an example of how to run the code:

```bash
python load_forecasting.py
```

## Requirements
Before running the code, ensure you have the required libraries installed:
```bash
pip install numpy pandas plotly scikit-learn tensorflow
```

## Learn More
To further explore deep learning models and data visualization, you can refer to the following resources:

- [TensorFlow Documentation](https://www.tensorflow.org/guide): Dive deeper into deep learning models and TensorFlow's capabilities.
- [Plotly Documentation](https://plotly.com/python/): Explore advanced data visualization techniques using Plotly.
