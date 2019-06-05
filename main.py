from neural import *
dataa = [
    [1, 1],
    [1, 0],
    [0, 0],
    [0, 1],
]
label = [
    [0],
    [1],
    [0],
    [1]
]
gg = Neural(2, [10], 1, .3)
for value in range(1000):
    gg.train(dataa, label, True)
print(gg.feedforward(dataa, True))
