########################################################
# xorNeuralNetworkKeras.py                             #
# Made by Teddy Etoeharnowo for Leiden University 2018 #
########################################################
import numpy
import keras
from keras.optimizers import SGD
from keras.layers import Dense
from keras.models import Input
from keras.models import Model
# Fixed random seed for reproducibility
numpy.random.seed(1383353)
# Import xorData-n dataset. Split into input (X) and output (Y) variables
dataset = numpy.loadtxt("xorData-10000.txt", delimiter=",")
X = dataset[:,0:2]
Y = dataset[:,2]
# Create model. 
# In this example: 2 inputs, a hidden layer with 2 neurons and an output layer with 1 output.
# TODO change number of neurons in hidden layer, change activation function.
visible = Input(shape=(2,))
hidden1 = Dense(2, activation='sigmoid')(visible)
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=visible, outputs=output)x
# Stochastic gradient descent optimizer.
# Parameters are: Learning rate, learning rate decay, momentum and whether to apply Nesterov momentum
# TODO adjust the learning rate and ...
sgd = SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
# Compile model for a mean squared error regression problem
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
# Train the model, iterating on the data in batches.
model.fit(X, Y, epochs=1, batch_size=1)
# Evaluate the model
scores = model.evaluate(X, Y)
print("\nResult is %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
