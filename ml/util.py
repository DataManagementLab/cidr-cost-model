import autograd.numpy as np


def concat_and_multiply(weights, *args):
    cat_state = np.hstack(args + (np.ones((args[0].shape[0], 1)),))
    return np.dot(cat_state, weights)


def sigmoid(x):
    return 0.5 * (np.tanh(x) + 1.0)  # Output ranges from 0 to 1.


def leaky_relu(input, weights):
    cat_state = np.hstack((input, np.ones((input.shape[0], 1))))
    result = np.dot(cat_state, weights)
    return np.where(result > 0, result, result * 0.01)


def figsize(scale=0.5, gm=3):  # default golden mean =3.5
    fig_width_pt = 497.92325  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / gm  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean * 1  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size
