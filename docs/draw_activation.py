import matplotlib.pyplot as plt
import numpy as np
import math


def sigmoid_function(z):
    return [1 / (1 + math.exp(-num)) for num in z]


def tanh_function(z):
    return [math.tanh(num) for num in z]


def leaky_relu_function(z, alpha=0.1):
    return [max(num, alpha * num) for num in z]


def mish_function(zs):
    return [z * math.tanh(math.log(1 + math.exp(z))) for z in zs]


if __name__ == '__main__':
    z = np.arange(-10, 10, 0.01)
    fzes = [sigmoid_function(z), tanh_function(z), leaky_relu_function(z), mish_function(z)]
    titles = ['(a) Sigmoid', '(b) Tanh', "(c) Leaky ReLU", "(d) Mish"]

    plt.figure(figsize=(10, 8), dpi=100)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.98, wspace=0.3, hspace=0.3)
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        axes = plt.gca()
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)
        plt.title(titles[i], y=-0.25)
        plt.xticks(np.linspace(-10, 10, 5))
        plt.xlabel('z')
        plt.ylabel('g(z)')
        plt.plot(z, fzes[i])
    plt.show()
    # plt.savefig("docs/activation functions.png")
