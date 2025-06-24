import numpy as np
from nnfs.datasets import spiral_data, sine_data
import nnfs
from tqdm import tqdm

nnfs.init()


X, y = spiral_data(samples=1000,  classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)
# Instantiate the model
model = Model()
# Add layers
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Activation_ReLU())
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())
# Set loss and optimizer objects
model.set(
    loss=Loss_CategoricalCrossEntropy(),
    optimizer=Optimizer_Adam(learning_rate=0.005, decay=5e-5),
    accuracy=Accuracy_Categorical()
)

model.finalize()

model.train(X, y, epochs=10000, print_every=100)
