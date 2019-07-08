import numpy as np
import random


class Matrix:
    def __init__(self, x, y):
        self.row = x
        self.col = y
        self.matrox = np.arange(x * y, dtype=np.float64).reshape(x, y)

    def randomize(self):
        for x in range(self.row):
            for y in range(self.col):
                self.matrox[x][y] = random.random() * 2 - 1

    def transpose(self):
        r = Matrix(self.col, self.row)
        r.matrox = self.matrox.T
        return r

    def add(self, mat):
        if mat.row == self.row and mat.col == self.col:
            self.matrox += mat.matrox
        else:
            print("not compatible")
            np.append()

    def map(self, func):
        for x in range(self.row):
            for y in range(self.col):
                self.matrox[x][y] = func(self.matrox[x][y])

    def toArray(self):
        return self.matrox.copy().flatten()
#----------------------------all the static methods of the matrix object-----------------------------------
    @staticmethod
    def dotProduct(xx, yy):
        try:
            tr = np.dot(xx.matrox, yy.matrox)
            r = Matrix(tr.shape[0], tr.shape[1])
            r.matrox = tr
            return r
        except Exception as e:
            print("dimention are not compatible static")

    @staticmethod
    def fromArray(arr):
        inppp = np.copy(arr).flatten()
        tab = inppp.reshape(inppp.shape[0], 1)
        inppp = Matrix(inppp.shape[0], 1)
        inppp.matrox = tab
        return inppp

    @staticmethod
    def sub(x, y):
        if x.row == y.row and x.col == y.col:
            r = Matrix(x.row, x.col)
            tab = x.matrox - y.matrox
            r.matrox = tab
            return r

    @staticmethod
    def Smap(selfii, func):
        selfi = Matrix(selfii.row, selfii.col)
        selfi.matrox = selfii.matrox.copy()
        selfi.matrox = func(selfi.matrox)
        return selfi

    @staticmethod
    def multElement(xx, yy):
        if xx.row == yy.row and xx.col == yy.col:
            tt = Matrix(xx.row, xx.col)
            tt.matrox = np.multiply(xx.matrox, yy.matrox)
            return tt
        else:
            print("not compatible multElement")
