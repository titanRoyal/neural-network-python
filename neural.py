import numpy as np
import math
from matrix import *


class Neural:
    def __init__(self, x, y, z, lr=0.055):
        self.inputNum = x
        self.hiddenNum = len(y)
        self.outputNum = z
        self.lr = lr
        self.weights = []
        self.bias = []
        for tt in range(self.hiddenNum + 1):
            if tt == 0:
                self.weights.append(Matrix(y[0], self.inputNum))
            elif tt == self.hiddenNum:
                self.weights.append(Matrix(self.outputNum, y[-1]))
            else:
                self.weights.append(Matrix(y[tt], y[tt - 1]))
            self.weights[tt].randomize()
        for tt in range(self.hiddenNum + 1):
            if tt == self.hiddenNum:
                self.bias.append(Matrix(z, 1))
            else:
                self.bias.append(Matrix(y[tt], 1))
            self.bias[tt].randomize()

    def feedforward(self, xy):
        inpp = Matrix.fromArray(xy)
        for tt in range(len(self.weights)):
            sum = Matrix.dotProductt(self.weights[tt], inpp)
            sum.add(self.bias[tt])
            sum.map(sigmoid)
            inpp = sum
        return inpp.toArray()

    def train(self, data, label):
        info = Matrix.fromArray(data)
        lab = Matrix.fromArray(label)
        sum_l = np.array([info])
        for value in range(len(self.weights)):
            sum_l[value] = Matrix.dotProductt(
                self.weights[value], sum_l[value])
            sum_l[value].add(self.bias[value])
            sum_l[value].map(sigmoid)
            sum_l = np.append(sum_l, sum_l[value])
        sum_l = np.delete(sum_l, -1)
        output_error = Matrix.sub(lab, sum_l[-1])
        err_tab = []
        for value in range(len(self.weights) - 1, -1, -1):
            if value == len(self.weights) - 1:
                err_tab.append(Matrix.dotProductt(
                    self.weights[value].transpose(), output_error))
            else:
                err_tab.append(Matrix.dotProductt(
                    self.weights[value].transpose(), err_tab[-1]))
        err_tab = err_tab[::-1]
        gradient = []
        deltaw = []
        for value in range(len(self.weights)):
            if value == 0:
                gradient.append(Matrix.mapp(sum_l[value], Dsigmoid))
                gradient[value] = Matrix.multElement(
                    gradient[value], err_tab[value + 1])
                gradient[value].matrox *= self.lr
                self.bias[value].add(gradient[value])
                deltaw.append(Matrix.dotProductt(
                    gradient[value], Matrix.transpose(info)))
            elif value == len(self.weights) - 1:
                gradient.append(Matrix.mapp(sum_l[value], Dsigmoid))
                gradient[value] = Matrix.multElement(
                    gradient[value], output_error)
                gradient[value].matrox *= self.lr
                self.bias[value].add(gradient[value])
                deltaw.append(Matrix.dotProductt(
                    gradient[value], Matrix.transpose(sum_l[value - 1])))
            else:
                gradient.append(Matrix.mapp(sum_l[value], Dsigmoid))
                gradient[value] = Matrix.multElement(
                    gradient[value], err_tab[value + 1])
                gradient[value].matrox *= self.lr
                self.bias[value].add(gradient[value])
                deltaw.append(Matrix.dotProductt(
                    gradient[value], Matrix.transpose(sum_l[value - 1])))
            self.weights[value].add(deltaw[value])


def sigmoid(x):
    return 1 / (1 + math.exp(x))


def Dsigmoid(x):
    return x * (1 - x)
