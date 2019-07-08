# Importing the Neural network library
from neural import *
# Initializ the training data
data = [
    [0, 1],
    [0, 0],
    [1, 1],
    [1, 0]
]
label = [
    [1],
    [0],
    [0],
    [1]
]
# Fiting the model
model = Neural(2, [10], 1, .3)
# Print the history of loss
log = model.train_batch(data, label, 3000)
print(log)
# Geting the new prediction from the trained neural network
print(model.feedforward(data, True))
