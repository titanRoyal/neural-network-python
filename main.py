from neural import *
import random
from math import *

data = [
    [1, 1],
    [0, 0],
    [1, 0],
    [0, 1]
]
label = [
    [0],
    [0],
    [1],
    [1]
]
gg = Neural(2, [6], 1, .4)
for value in range(1000):
    index = random.randint(0, 3)
    gg.train(data[index], label[index])

print(gg.feedforward(data,label))
