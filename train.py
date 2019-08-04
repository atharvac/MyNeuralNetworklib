import tensorflow as tf
from Neural2 import *
import matplotlib.pyplot as plt
#from matplotlib.image import imread
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0

def gen_target(n):
    return np.array([0 if x!=n else 1 for x in range(10)])

def main():
    nn = NeuralNetwork(784, 400, 10)
    for y in range(3):
        print(f"Start {y+1} epoch")
        for x in range(len(x_train)):
            if x%20 == 0:
                print(x)
            nn.train(x_train[x].reshape(784,), gen_target(y_train[x]), 0.01)

    nn = NeuralNetwork.save("mnist")

    for x in range(10):
        print(np.argmax(nn.guess(x_train[x].reshape(784,))))
        plt.imshow(x_train[x])
        plt.show()




main()
#nn = NeuralNetwork()
