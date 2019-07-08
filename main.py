from neural import *
dataa = [
    [1, 1],
    [1, 0],
    [0, 0],
    [0, 1],
]
label = [
    [.2,.5],
    [.5,.7],
    [.7,1],
    [1,.2]
]
gg = Neural(2, [10], 2, .3)
for value in range(5000):
    gg.train(dataa, label, True)
print(gg.feedforward(dataa, True))
