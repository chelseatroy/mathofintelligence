import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from ui_tools import colors
from ui_tools import regression_plot_with_feature_weights

# MATH OF INTELLIGENCE WEEK 1 ASSIGNMENT

# TODO:
# add the option to step until the total cost is below a certain threshold (how to decide?)
# add the option to step until the gradient is below a certain threshold for all features (how to decide?)
# add documentation

class GradientDescent():
    weights = []
    intercept = 0

    x = []
    y = []

    maybe_labels = []

    dataframe_data = None

    def fit(self, x, y):
        if isinstance(x, np.ndarray):
            self.x = x.tolist()
            self.y = y.tolist()
            number_of_features = x.shape[0]
            number_of_points = x.shape[1]
        elif isinstance(x, pd.DataFrame):
            self.dataframe_data = x
            self.maybe_labels = x.columns.values.tolist()
            self.x = self.feature_arrays_from(x)
            self.y = y.values
            number_of_features = x.shape[1]
            number_of_points = x.shape[0]

        if number_of_points != len(self.y): raise Exception(
            "Number of feature values not equal to number of output values.")

        num_iterations = 1000
        self.weights = [0 for _ in range(number_of_features)]

        for i in range(num_iterations):
            self.step()

    def feature_arrays_from(self, x):
        feature_array = []
        for column in x:
            x_array = x[column]
            feature_array.append(x_array)
        return feature_array

    def predict(self, input_data):
        predicted_values = [self.intercept for _ in range(len(input_data[0]))]

        for feature_index, feature in enumerate(input_data):
            for point_index, point in enumerate(feature.tolist()):
                predicted_values[point_index] += self.weights[feature_index] * point

        return predicted_values

    def plot(self):
        if len(self.x) == 1:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.scatter(self.x[0], self.y)
            ax.plot(self.x[0], self.predict(self.x))
            fig.savefig('plot.png')
        else:
            regression_plot_with_feature_weights(self.dataframe_data, self.y, self.weights, self.predict(self.x),
                                                     self.intercept)

    def step(self, learning_rate=0.0001):
        for index, feature in enumerate(self.x):
            feature_gradient = 0
            intercept_gradient = 0
            for i in range(len(feature)):
                N = float(len(feature))
                x_val = feature[i]
                y_val = self.y[i]
                intercept_gradient += -(2 / N) * (y_val - ((self.weights[index] * x_val) + self.intercept))
                feature_gradient += -(2/N) * x_val * (y_val - ((self.weights[index] * x_val) + self.intercept))
            self.intercept = self.intercept - (learning_rate * intercept_gradient)
            self.weights[index] = self.weights[index] - (learning_rate * feature_gradient)

    def calculate_cost(self):
        squared_cost = 0
        predictions = self.predict(self.x)

        for index, outcome in enumerate(self.y):
            squared_cost += (outcome - predictions[index]) **2

        cost = squared_cost**(0.5)

        return cost

g = GradientDescent()
data = pd.read_csv('data.csv')

x_values = data[['some_feature', 'some_other_feature', 'yet_another_feature', 'a_fourth_feature']]
y_values = data['value_to_predict']
g.fit(x_values, y_values)
g.plot()


