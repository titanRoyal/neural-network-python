import numpy as np
import math
from matrix import *


class Neural:
    def __init__(self, x, y, z, lr=0.055):
        self.inputNum = x
        self.hiddenNum = y
        self.outputNum = z
        self.hiddenLayer = len(y)
        self.lr = lr
        self.weights = []
        self.bias = []
        for tt in range(self.hiddenLayer + 1):
            if tt == 0:
                self.weights.append(Matrix(y[0], self.inputNum))
            elif tt == self.hiddenLayer:
                self.weights.append(Matrix(self.outputNum, y[-1]))
            else:
                self.weights.append(Matrix(y[tt], y[tt - 1]))
            self.weights[tt].randomize()
        for tt in range(self.hiddenLayer + 1):
            if tt == self.hiddenLayer:
                self.bias.append(Matrix(z, 1))
            else:
                self.bias.append(Matrix(y[tt], 1))
            self.bias[tt].randomize()

    def feedforward(self, data, isBatch=False):
        if not(isBatch):
            sumi = Matrix.fromArray(data)
            for tt in range(len(self.weights)):
                sumi = Matrix.dotProduct(self.weights[tt], sumi)
                sumi.add(self.bias[tt])
                sumi.map(sigmoid)
                # target = sumi
            return sumi.toArray()
        else:
            output = np.copy([])
            for value in data:
                sumi = Matrix.fromArray(value)
                for tt in range(len(self.weights)):
                    sumi = Matrix.dotProduct(self.weights[tt], sumi)
                    sumi.add(self.bias[tt])
                    sumi.map(sigmoid)
                    # target = sumi
                output = np.append(output, sumi.toArray())
            return output.reshape(len(data), self.outputNum)

    def train(self, data, label, isBatch=False):
        if not(isBatch):
            info = Matrix.fromArray(data)
            lab = Matrix.fromArray(label)
            sum_l = [info]
            for value in range(len(self.weights)):
                sum_l[value] = Matrix.dotProduct(
                    self.weights[value], sum_l[value])
                sum_l[value].add(self.bias[value])
                sum_l[value].map(sigmoid)
                sum_l.append(sum_l[value])
            sum_l.remove(sum_l[-1])
            output_error = Matrix.sub(lab, sum_l[-1])
            err_tab = []
            for value in range(len(self.weights) - 1, -1, -1):
                if value == len(self.weights) - 1:
                    err_tab.append(Matrix.dotProduct(
                        self.weights[value].transpose(), output_error))
                else:
                    err_tab.append(Matrix.dotProduct(
                        self.weights[value].transpose(), err_tab[-1]))
            err_tab = err_tab[::-1]
            err_tab.append(output_error)
            for value in range(len(self.weights)):
                gradient = Matrix.Smap(sum_l[value], Dsigmoid)
                gradient = Matrix.multElement(
                    gradient, err_tab[value + 1])
                gradient.matrox *= self.lr
                if value == 0:
                    deltaw = Matrix.dotProduct(
                        gradient, info.transpose())
                else:
                    deltaw = Matrix.dotProduct(
                        gradient, sum_l[value - 1].transpose())
                self.bias[value].add(gradient)
                self.weights[value].add(deltaw)

        else:
            for value1 in range(len(data)):
                info = Matrix.fromArray(data[value1])
                lab = Matrix.fromArray(label[value1])
                sum_l = [info]
                for value in range(len(self.weights)):
                    sum_l[value] = Matrix.dotProduct(
                        self.weights[value], sum_l[value])
                    sum_l[value].add(self.bias[value])
                    sum_l[value].map(sigmoid)
                    sum_l.append(sum_l[value])
                sum_l.remove(sum_l[-1])
                output_error = Matrix.sub(lab, sum_l[-1])
                err_tab = []
                for value in range(len(self.weights) - 1, -1, -1):
                    if value == len(self.weights) - 1:
                        err_tab.append(Matrix.dotProduct(
                            self.weights[value].transpose(), output_error))
                    else:
                        err_tab.append(Matrix.dotProduct(
                            self.weights[value].transpose(), err_tab[-1]))
                err_tab = err_tab[::-1]
                err_tab.append(output_error)
                for value in range(len(self.weights)):
                    gradient = Matrix.Smap(sum_l[value], Dsigmoid)
                    gradient = Matrix.multElement(
                        gradient, err_tab[value + 1])
                    gradient.matrox *= self.lr
                    if value == 0:
                        deltaw = Matrix.dotProduct(
                            gradient, info.transpose())
                    else:
                        deltaw = Matrix.dotProduct(
                            gradient, sum_l[value - 1].transpose())
                    self.bias[value].add(gradient)
                    self.weights[value].add(deltaw)

            print("loss: " + str(abs(np.copy(label) -
                                     np.copy(self.feedforward(data, True))).sum() / self.outputNum))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def Dsigmoid(x):
    return x
