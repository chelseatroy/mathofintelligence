import numpy as np
import matplotlib.pyplot as plt

# TODO:
# add the option to step until the total cost is below a certain threshold
# add the option to step until the gradient is below a certain threshold
# add documentation
# make multi-feature bar chart visualizations

class GradientDescent():
    weights = []
    intercept = 0

    x = []
    y = []

    def fit(self, x, y):
        self.x = x
        self.y = y
        number_of_features = x.shape[0]
        number_of_points = x.shape[1]

        if number_of_points != len(self.y): raise Exception(
            "Number of feature values not equal to number of output values.")

        num_iterations = 1000
        self.weights = [0 for _ in range(number_of_features)]

        for i in range(num_iterations):
            self.step(x, y)
            if i % 100 == 0:
                print("weights: ")
                print(self.weights)
                print("intercept: ")
                print(self.intercept)

    def predict(self, input_data):
        predicted_values = [self.intercept for _ in range(len(input_data[0]))]

        for feature_index, feature in enumerate(input_data):
            for point_index, point in enumerate(feature.tolist()):
                predicted_values[point_index] += self.weights[feature_index] * point

        return predicted_values

    def plot(self):
        if len(self.x) == 1:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            print("x")
            print(self.x)
            ax.scatter(self.x[0], self.y)
            ax.plot(self.x[0], self.predict(self.x))
            fig.savefig('points.png')
        else:
            self.bar_chart_bonanzaaaaa()

    def bar_chart_bonanzaaaaa(self):
        _ = 7

    def step(self, x, y, learning_rate=0.0001):
        for index, feature in enumerate(x.tolist()):
            feature_gradient = 0
            intercept_gradient = 0
            for i in range(len(feature)):
                N = float(len(feature))
                x_val = feature[i]
                y_val = y[i]
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
x_values = np.asarray([[1, 2, 3, 5, 7, 8, 9, 5, 6, 7, 3, 5, 6, 7, 2, 5, 6, 8]])
y_values = np.asarray([1, 2, 4, 5, 6, 7, 2, 5, 7, 8, 2, 7, 3, 2, 6, 1, 7, 2])
g.fit(x_values, y_values)
g.plot()


