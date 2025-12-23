import numpy as np

class Network:
    '''
    Implementation of a neural network that can be used as an Autoencoder
    '''

    def __init__(self, sizes, alpha, iterations, batch_size):
        '''
        Network constructor
        
        :param sizes: Dimensions [X, X->Y, Y->Z]
        :param alpha: Learning rate
        :param iterations: Iterations
        :param batch_size: Batch size
        '''
        
        self.weights = {}
        self.biases = {}
        for i in range(len(sizes)-1):
            # optionally add * np.sqrt(1./sizes[i]) to shrink variance
            self.weights[i] = np.random.randn(sizes[i+1], sizes[i]) * np.sqrt(1./sizes[i+1])
            self.biases[i] = np.array([np.full(sizes[i+1], 0.25)])

        self.alpha = alpha
        self.iterations = iterations
        self.batch_size = batch_size
        self.sizes = sizes

    def _loss(self, y, yhat):
        '''
        Squared error loss function
        
        :param y: Label
        :param yhat: Prediction
        '''
        return np.sum(0.5 * (y - yhat)**2), y - yhat
    
    def _activation(self, x):
        '''
        Sigmoid activation function and first it's derivative for backpropagation
        
        :param x: Weights*Input+Bias
        '''
        f = 1 / (1 + np.exp(-x))
        df = f * (1 - f)
        return f, df
    
    def _feedforward(self, X):
        '''
        Propagates the input in forward direction
        
        :param X: Input as row vector or a matrix with 1 column
        '''
        self._z = {}
        self._a = {0: X}

        for i in range(len(self.weights)):
            self._z[i] = self.weights[i] @ self._a[i] + self.biases[i].T
            self._a[i+1] = self._activation(self._z[i])[0]
        return self._a[i+1]
    
    def _backpropagation(self, y, yhat):
        '''
        Backpropagation algorithm
           
        :param y: Label
        :param yhat: Prediction
        '''
        n = len(self.weights)
        # remove -1 to use the positive gradient, which will result in inverted colors
        delta = -1 * self._loss(y, yhat)[1] * self._activation(self._z[n-1])[1]
        grad_weights = { n-1: delta @ self._a[n-1].T}
        grad_biases = { n-1: delta }

        for i in reversed(range(n-1)):
            delta = self.weights[i+1].T @ delta * self._activation(self._z[i])[1]
            grad_weights[i] = delta @ self._a[i].T
            grad_biases[i] = delta

        return grad_weights, grad_biases

    def train(self, X, y):
        '''
        Model training with stochastic gradient descent
        
        :param X: Input
        :param y: Label
        '''
        X_batch = X
        y_batch = y
        
        for _ in range(self.iterations):
            if self.batch_size > 0 and self.batch_size < X.shape[1]:
                k = np.random.choice(range(X.shape[1]), self.batch_size, replace=False)
                X_batch = X[k,:]
                y_batch = y[k,:]
            
            y_hat = self._feedforward(X_batch)
            grad_weights, grad_biases = self._backpropagation(y_batch, y_hat)

            for j in range(len(self.weights)):
                self.weights[j] -= self.alpha * grad_weights[j]
                self.biases[j] -= self.alpha * grad_biases[j].T

    def predict(self, X):
        '''
        Predicts the given input
        
        :param X: Input as row vector or a matrix with 1 column
        '''
        a = X

        for i in range(len(self.weights)):
            z = self.weights[i] @ a + self.biases[i].T
            a = self._activation(z)[0]
        return a

    def print(self):
        '''
        Prints all Network settings
        
        '''
        print('Iterations: ' + str(self.iterations))
        print('Learning Rate: ' + str(self.alpha))
        print('Batch Size: ' + str(self.batch_size))
        for i in range(len(self.sizes)-1):
            print('Weights at Layer ' + str(i+1) + ':')
            print(self.weights[i])
            print('Biases at Layer ' + str(i+1) + ':')
            print(self.biases[i])