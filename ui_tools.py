import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.legend_handler import HandlerPatch

def make_legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return mpatches.Circle(xy=(0.5*width-0.5*xdescent, 0.5*height-0.5*ydescent), radius = 4)

#PLEASE NORMALIZE YOUR MODEL BEFORE USING THIS. You should do it anyway, but in this case it makes it much easier
#to compare importance of features based on their bar size
#Actual values show up as dots. Predicted values show up as red Xs.
def regression_plot_with_feature_weights(dataframe_data, y_values, weights, predictions, intercept, filename="SMANGITY"):
    label_colors = []
    legend_patches = []

    number_of_data_points = dataframe_data.shape[0]
    legend_labels = dataframe_data.columns.values.tolist()

    indices = np.arange(number_of_data_points)
    intercepts = np.full((number_of_data_points,), intercept, )

    dataframe_data["predictions"] = predictions
    dataframe_data["outcomes"] = y_values
    dataframe_data["intercept"] = intercepts.tolist()
    dataframe_data.sort_values("outcomes", inplace=True)

    sorted_predictions = dataframe_data["predictions"]
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
    ax.scatter(indices, sorted_predictions, s=50, c='r', marker='_')
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
    for i, label in enumerate(legend_labels):
        legend_patch = mpatches.Patch(color=label_colors[i] + '70', label=label)
        legend_patches.append(legend_patch)
        final_index = i
    legend_patches.append(mpatches.Patch(color=label_colors[final_index + 1] + '70', label='intercept'))
    legend_patches.append(mpatches.Circle((1.04, 1), 0.1, label='outcomes'))
    legend_patches.append(mpatches.Rectangle((1.04, 1), width=5, height=0.1, label='predictions', color='r'))

    plt.subplots_adjust(right=0.7)
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.04,1), loc="upper left", handler_map={mpatches.Circle:HandlerPatch(patch_func=make_legend_circle),})

    plt.savefig(filename + '.png')

def feature_arrays_from(x):
    feature_array = []
    for column in x:
        x_array = x[column]
        feature_array.append(x_array)
    return feature_array

colors = [
'#FF0000',
'#E2571E',
'#FF7F00',
'#FFFF00',
'#00FF00',
'#96bf33',
'#0000FF',
'#4B0082',
'#8B00FF',
"#006400",
"#BDB76B",
"#8B008B",
"#556B2F",
"#FF8C00",
"#9932CC",
"#8B0000",
"#E9967A",
"#8FBC8F",
"#483D8B",
"#2F4F4F",
"#2F4F4F",
"#00CED1",
"#9400D3",
"#FF1493",
"#00BFFF",
"#696969",
"#696969",
"#1E90FF",
"#B22222",
"#FFFAF0",
"#228B22",
"#FF00FF",
"#DCDCDC",
"#F8F8FF",
"#FFD700",
"#DAA520",
"#808080",
"#808080",
"#008000",
"#ADFF2F",
"#F0FFF0",
"#FF69B4",
"#CD5C5C",
"#4B0082",
"#FFFFF0",
"#F0E68C",
"#E6E6FA",
"#FFF0F5",
"#7CFC00",
"#FFFACD",
"#ADD8E6",
"#F08080",
"#E0FFFF",
"#FAFAD2",
"#D3D3D3",
"#D3D3D3",
"#90EE90",
"#FFB6C1",
"#FFA07A",
"#20B2AA",
"#87CEFA",
"#778899",
"#778899",
"#B0C4DE",
"#FFFFE0",
"#00FF00",
"#32CD32",
"#FAF0E6",
"#FF00FF",
"#800000",
"#66CDAA",
"#0000CD",
"#BA55D3",
"#9370DB",
"#3CB371",
"#7B68EE",
"#00FA9A",
"#48D1CC",
"#C71585",
"#191970",
"#F5FFFA",
"#FFE4E1",
"#FFE4B5",
"#FFDEAD",
"#000080",
"#FDF5E6"
]