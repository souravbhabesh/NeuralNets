import random
import numpy as np

class DNN(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[1:],sizes[:-1])]

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoidPrime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feedForward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # training_data = list(training_data)
        n = len(training_data)
        for e in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for batch in mini_batches:
                self.update_mini_batch(batch, eta)
            if test_data:
                test_data = list(test_data)
                n_test = len(test_data)
                print("Epoch {} : {} / {}".format(e, self.evaluate(test_data), n_test));
            else:
                print("Epoch {} complete".format(e))
        return

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backPropagation(x,y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backPropagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feed forward step
        activation = x
        activations = [x]
        # list to maintain the z values at each layer
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # Back propagation
        # BP 1
        delta = self.costDerivative(activations[-1], y) * self.sigmoidPrime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoidPrime(z)
            delta = np.multiply(np.dot(self.weights[-l+1].transpose(), delta), sp)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedForward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    def costDerivative(self, output_activations, y):
        return output_activations-y



if __name__ == '__main__':
    net = DNN(sizes = [2, 3, 1])
    print(net.biases[0].shape)
    print(net.weights[0].shape)
    print(net.weights[1].shape)