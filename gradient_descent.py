import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# TODO:
# have the plot sort by increasing predict() value and distribute along x axis by point index
# add the option to step until the total cost is below a certain threshold
# add the option to step until the gradient is below a certain threshold
# add documentation

class GradientDescent():
    weights = []
    intercept = 0

    x = []
    y = []
    maybe_labels = []

    def fit(self, x, y):
        if isinstance(x, np.ndarray):
            self.x = x.tolist()
            self.y = y.tolist()
            number_of_features = x.shape[0]
            number_of_points = x.shape[1]
        elif isinstance(x, pd.DataFrame):
            self.maybe_labels = x.columns.values.tolist()
            for column in x:
                x_array = x[column]
                self.x.append(x_array)
            self.y = y.values
            number_of_features = x.shape[1]
            number_of_points = x.shape[0]

        if number_of_points != len(self.y): raise Exception(
            "Number of feature values not equal to number of output values.")

        num_iterations = 1000
        self.weights = [0 for _ in range(number_of_features)]

        for i in range(num_iterations):
            self.step()

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
            self.plot_with_feature_weights()

    def plot_with_feature_weights(self):
        colors = ['#B000B5', '#C0FFEE', '#BADA55', '#D11D05', '#C0FF33', '#10ADED']

        number_of_data_points = len(self.x[0])
        indices = np.arange(number_of_data_points)
        intercepts = np.full((number_of_data_points,), self.intercept)
        height_so_far = intercepts
        color_index_changeme = 0

        plt.bar(indices, intercepts, 0.3, color='#d62728', label="intercept")

        for feature_index, feature in enumerate(self.x):
            weight = self.weights[feature_index]
            weight_account = np.array(feature) * weight

            color = colors[color_index_changeme]
            plt.bar(indices, weight_account, 0.3, color=color, bottom=height_so_far, label= self.maybe_labels[feature_index] or "feature %d" % (feature_index))
            height_so_far += weight_account
            color_index_changeme += 1

        plt.legend()
        plt.savefig('poonts.png')

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
        cost = 0

        for index, outcome in enumerate(self.y):
            factors = 0
            for feature in self.x:
                factors += feature[index] * self.weights[index]
            cost += (outcome - factors) **2

        return cost

g = GradientDescent()
data = pd.read_csv('data.csv')

x_values = data[['some_feature', 'some_other_feature']]
y_values = data['value_to_predict']
g.fit(x_values, y_values)
g.plot()


