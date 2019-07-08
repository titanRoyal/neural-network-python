from neural import *
data = [
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
model = Neural(2, [10], 1, .3)
lomodel=model.train_batch(data, label,2000)
print(lomodel)
print(model.feedforward(data, True))
