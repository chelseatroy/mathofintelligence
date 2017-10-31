import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

colors = ['#EFFEC7', '#C0FFEE', '#CC5555', '#BADA55', '#EB01A5', '#C0FF33', '#10ADED']

def regression_plot_with_feature_weights(dataframe_data, x_values, y_values, weights, predictions, intercept, maybe_labels):
    label_colors = []
    legend_patches = []

    number_of_data_points = len(x_values[0])
    indices = np.arange(number_of_data_points)
    intercepts = np.full((number_of_data_points,), intercept, )

    dataframe_data["predictions"] = predictions
    dataframe_data["outcomes"] = y_values
    dataframe_data["intercept"] = intercepts.tolist()
    dataframe_data.sort_values("predictions", inplace=True)
    dataframe_data.drop('predictions', axis=1, inplace=True)
    y_values = dataframe_data['outcomes'].values
    dataframe_data.drop('outcomes', axis=1, inplace=True)
    feature_arrays_from(dataframe_data)
    weights.append(1)
    for i in range(0, len(weights)):
        dataframe_data.ix[:, i] = dataframe_data.ix[:, i] * weights[i]
    x_values = feature_arrays_from(dataframe_data)

    ax = plt.subplot(111)
    ax.scatter(indices, y_values, s=70)
    ax.axhline(0, color="gray")

    data_with_weights = np.asarray(x_values)

    data_shape = np.shape(data_with_weights)

    # Take negative and positive data apart and cumulate
    def get_cumulated_array(data, **kwargs):
        cum = data.clip(**kwargs)
        cum = np.cumsum(cum, axis=0)
        d = np.zeros(np.shape(data))
        d[1:] = cum[:-1]
        return d

    cumulated_data = get_cumulated_array(data_with_weights, min=0)
    cumulated_data_neg = get_cumulated_array(data_with_weights, max=0)

    # Re-merge negative and positive data.
    row_mask = (data_with_weights < 0)
    cumulated_data[row_mask] = cumulated_data_neg[row_mask]
    data_stack = cumulated_data

    for i in np.arange(0, data_shape[0]):
        ax.bar(np.arange(data_shape[1]), data_with_weights[i], bottom=data_stack[i], color=colors[i], alpha=0.3)
        label_colors.append(colors[i])

    final_index = 0
    for i, label in enumerate(maybe_labels):
        legend_patch = mpatches.Patch(color=label_colors[i] + '70', label=label)
        legend_patches.append(legend_patch)
        final_index = i
    legend_patches.append(mpatches.Patch(color=label_colors[final_index + 1] + '70', label='intercept'))

    plt.legend(handles=legend_patches)

    plt.savefig('regression_plot_with_weights.png')

def feature_arrays_from(x):
    feature_array = []
    for column in x:
        x_array = x[column]
        feature_array.append(x_array)
    return feature_array