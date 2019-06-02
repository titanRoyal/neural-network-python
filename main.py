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
gg = Neural(2, [6], 5, .4)
for value in range(1000):
    index = random.randint(0, 3)
    gg.train([1,1], [1,.75,.5,.25,0])

print(gg.feedforward(data[0]))
print(gg.feedforward(data[1]))
print(gg.feedforward(data[2]))
print(gg.feedforward(data[3]))
