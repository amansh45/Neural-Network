import numpy as np
import math
import copy
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import softmax
from scipy.special import expit
import time
from matplotlib import pyplot as plt

class Layer:
    def __init__(self, num_neurons = None, activation = None, batch_size = None):
        self.num_neurons = num_neurons
        self.activation = activation
        self.input = np.zeros((batch_size, num_neurons))
        self.output = np.zeros((batch_size, num_neurons))
        self.delta = np.zeros((batch_size, num_neurons))

class Model:
    def __init__(self, batch):
        self.layers = []
        self.weights = {}
        self.weight_labels = []
        self.loss = None
        self.learn_rate = None
        self.batch = batch

    def initialize_weights(self, input_layer_len):
        self.num_weights = len(self.layers)
        num_neurons = []
        for layer in self.layers:
            num_neurons.append(layer.num_neurons)
        weight_mat_dimensions = []
        weight_mat_dimensions.append([input_layer_len, num_neurons[0]])
        self.weight_labels.append("i_h0")
        for i in range(len(num_neurons)-1):
            weight_mat_dimensions.append([num_neurons[i], num_neurons[i+1]])
            self.weight_labels.append("h"+str(i)+"_h"+str(i+1))
            
        for i in range(len(weight_mat_dimensions)):
            self.weights[self.weight_labels[i]] = np.random.rand(weight_mat_dimensions[i][0],weight_mat_dimensions[i][1])

        for i in range(len(weight_mat_dimensions)):
            self.weights[self.weight_labels[i]] = self.weights[self.weight_labels[i]] * math.sqrt(2.0/self.weights[self.weight_labels[i]].shape[0])


    def apply_activation(self, in_arr, activation):
        if activation == "sigmoid":
            return 1.0 / (1 + np.exp(-in_arr))
        elif activation == "relu":
            out_arr = np.maximum(0,in_arr)
            return out_arr
        elif activation == "softmax":
            return softmax(in_arr)
        elif activation == 'tanh':
            return np.tanh(in_arr)

    def derivative_of_activation(self, activation, in_data):
        derived_data = []
        if activation == "sigmoid":
            return self.apply_activation(in_data, 'sigmoid') * (1 - self.apply_activation(in_data, 'sigmoid'))
        elif activation == "relu":
            derived_data = copy.deepcopy(in_data)
            derived_data[derived_data <= 0] = 0
            derived_data[derived_data > 0] = 1
            return derived_data
        elif activation == "softmax":
            for x in in_data:
                x_data = []
                sqr_sum = 0
                for neuron in x:
                    sqr_sum += math.e**(neuron)
                sqr_sum = sqr_sum ** 2

                for i in range(len(x)):
                    num = math.e**(x[i])
                    num_sum = 0
                    for j in range(len(x)):
                        if i==j:
                            continue
                        else:
                            num_sum += math.e**(x[j])
                    num = num * num_sum
                    x_data.append(num/sqr_sum)
                derived_data.append(x_data)
            return derived_data
        elif activation == 'tanh':
            tanh_z = np.tanh(in_data)
            return 1 - (tanh_z)**2
    
    def add(self, neurons = None, activation = None):
        layer = Layer(num_neurons = neurons, activation = activation, batch_size = self.batch)
        self.layers.append(layer)

    def compile(self, loss = "categorical_crossentropy", learn_rate = 0.01):
        self.loss = loss
        self.learn_rate = learn_rate

    def calculate_loss(self, labels):
        if self.loss == "categorical_crossentropy":
            return (self.layers[-1].output - labels)

    def forward_propagate(self, input_sample):
        new_input = np.array([])
        for i, lab in enumerate(self.weight_labels):
            if i == 0:
                t = np.dot(input_sample, self.weights[lab])
            else:
                t = np.dot(new_input, self.weights[lab])
            self.layers[i].input = copy.deepcopy(t)
            self.layers[i].output = self.apply_activation(t, self.layers[i].activation)
            new_input = self.layers[i].output


    def back_propagation(self, sample, lab):
        i = len(self.layers)-1
        dE_dW = {}
        yminus_yhat = self.calculate_loss(lab)
        while i>=0:
            if i == len(self.layers)-1:
                weights_name = self.weight_labels[i]
                self.layers[i].delta = yminus_yhat
                exp_output = copy.deepcopy(self.layers[i-1].output)
                delta_weights = np.dot(exp_output.T, yminus_yhat)
                dE_dW[weights_name] = delta_weights
            else:
                weights_name = self.weight_labels[i]
                previous_delta = copy.deepcopy(self.layers[i+1].delta)
                din_dout = self.weights[self.weight_labels[i+1]]

                dout_din = self.derivative_of_activation(self.layers[i].activation, self.layers[i].input)

                dcost_dah = np.dot(previous_delta, din_dout.T)
                if i == 0:
                    din_dW = sample
                else:
                    din_dW = self.layers[i-1].output
                self.layers[i].delta = np.multiply(dcost_dah, dout_din)
                
                dE_dW[weights_name] = np.dot(din_dW.T, self.layers[i].delta)

            i-=1
        for j in range(len(self.layers)):
            weights_name = self.weight_labels[j]
            self.weights[weights_name] = self.weights[weights_name] - np.multiply(dE_dW[weights_name], self.learn_rate)
            

    def predict(self, test_x):
        self.forward_propagate(np.array(test_x))
        return self.layers[len(self.layers)-1].output

    def accuracy(self, f, s):
        true = 0
        for i in range(len(f)):
            if f[i] == s[i]:
                true += 1
        return true/len(f)

    def convert_encoding(self, labels, flag):
        if flag:
            new_labs = []
            for lab in labels:
                max_index = 0
                max_prob = lab[0]
                for i in range(len(lab)):
                    if lab[i] > max_prob:
                        max_prob = lab[i]
                        max_index = i
                new_labs.append(max_index)
            return np.array(new_labs)
        elif flag == False:
            new_labs = []
            for lab in labels:
                test_lab = np.zeros(len(np.unique(labels))).tolist()
                test_lab[int(lab)] = 1
                new_labs.append(test_lab)
            return np.array(new_labs)


    def fit(self, train_x, train_labels, epochs = 10):
        self.initialize_weights(len(train_x[0]))
        one_hat_vectors = self.convert_encoding(train_labels, False)
        print("Train samples count: ",len(train_x))

        for j in range(epochs):
            num_batches = math.ceil(len(train_x)/self.batch)
            for i in range(num_batches):
                batch = train_x[i*self.batch:(i+1)*self.batch]
                batch_lab = one_hat_vectors[i*self.batch:(i+1)*self.batch]
                self.forward_propagate(np.array(batch))
                self.back_propagation(np.array(batch), np.array(batch_lab))
            predictions = self.predict(train_x)
            predictions = self.convert_encoding(predictions, True)
            accuracy = self.accuracy(predictions, train_labels)
            print("Epoch count:", j," accuracy over training data:", accuracy)


def load_data():
    global lab_mean
    l = os.getcwd().split('/')
    l.pop()
    data_csv = '/'.join(l) + "/input_data/apparel-trainval.csv"
    raw_data = pd.read_csv(data_csv, header=0)
    raw_data = raw_data.astype('float64')
    complete_array = raw_data.values
    Y_array = complete_array[:, 0]
    X_array = np.delete(complete_array, (0), axis=1)

    X_array = (X_array - np.mean(X_array, axis=0)) / np.std(X_array, axis=0)

    split_size = 0.20
    seed = 1

    X_train, X_val, Y_train, Y_val = train_test_split(X_array, Y_array, test_size=split_size)
    return (X_train, Y_train) , (X_val, Y_val)

def load_test():
    l = os.getcwd().split('/')
    l.pop()
    data_csv = '/'.join(l) + "/input_data/apparel-test.csv"
    test_dataset = pd.read_csv(data_csv)
    test_dataset = test_dataset.values
    X_array = test_dataset.astype('float64')
    X_array = (X_array - np.mean(X_array, axis=0)) / np.std(X_array, axis=0)
    return X_array

train, validation = load_data()


m = Model(batch = 100)
m.add(neurons = 128, activation = "relu")
m.add(neurons = 10, activation = "softmax")
m.compile(loss = "categorical_crossentropy", learn_rate = 0.001)
m.fit(train[0], train[1], epochs = 5)

m.forward_propagate(validation[0])
val_predictions = m.layers[-1].output
val_predictions = m.convert_encoding(val_predictions, True)
print("Accuracy over the validation dataset is: ", m.accuracy(val_predictions, validation[1]))

