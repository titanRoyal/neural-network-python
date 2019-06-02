import numpy as np
import random


class Matrix:
    def __init__(self, x, y):
        self.row = x
        self.col = y
        self.matrox = np.zeros((x, y), dtype=float)

    def randomize(self):
        for x in range(self.row):
            for y in range(self.col):
                self.matrox[x][y] = random.random() * 2 - 1

    def dotProduct(self, x):
        try:
            self.matrox = self.matrox.dot(x.matrox)
        except:
            print("dimention are not compatible")

    def add(self, mat):
        if mat.row == self.row and mat.col == self.col:
            self.matrox += mat.matrox
        else:
            print("not compatible")

    # def elementDot(self, x):
    #     try:
    #         self.matrox = self.matrox * x.matrox
    #     except:
    #         print("dimention are not compatible")
    # # Static methods
    @staticmethod
    def dotProductt(xx, yy):
        try:
            tr = xx.matrox.dot(yy.matrox)
            r = Matrix(tr.shape[0], tr.shape[1])
            r.matrox = tr
            return r
        except Exception as e:
            print("dimention are not compatible static")
            print(e)

    def map(self, func):
        for x in range(self.row):
            for y in range(self.col):
                self.matrox[x][y] = func(self.matrox[x][y])

    @staticmethod
    def fromArray(arr):
        inpp = np.array(arr)
        tab = inpp.reshape(len(arr), 1)
        inpp = Matrix(len(arr), 1)
        inpp.matrox = tab
        return inpp

    def toArray(self):
        return self.matrox.flatten()

    @staticmethod
    def sub(x, y):
        if x.row == y.row and x.col == y.col:
            r = Matrix(x.row, x.col)
            tab = y.matrox - x.matrox
            r.matrox = tab
            return r

    def transpose(self):
        r = Matrix(self.col, self.row)
        tab = self.matrox.copy().T
        r.matrox = tab
        return r

    @staticmethod
    def mapp(selfii, func):
        selfi = Matrix(selfii.row, selfii.col)
        selfi.matrox = selfii.matrox.copy()
        for x in range(selfi.row):
            for y in range(selfi.col):
                selfi.matrox[x][y] = func(selfi.matrox[x][y])
        return selfi

    @staticmethod
    def multElement(xx, yy):
        if xx.row == yy.row and xx.col == yy.col:
            tt = Matrix(xx.row, xx.col)
            tt.matrox = xx.matrox * yy.matrox
            return tt
        else:
            print("not compatible multElement")
