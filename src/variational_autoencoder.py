import numpy as np
from .network import Network

class VAE:
    '''
    Variational Autoencoder implementation
    '''

    def __init__(self, sizes, latent_dim, alpha, iterations, batch_size):
        '''
        VAE constructor
        
        :param sizes: Dimensions of the encoder and the decoder -> [[X->Y], [Y->Z]]
        :param latent_dim: Dimension of latent space
        :param alpha: Learning rate
        :param iterations: Iterations
        :param batch_size: Batch size
        '''
        
        self.latent_dim = latent_dim
        # add a layer of two neurons at the end to which will represent the encoded parameters
        self.encoder = Network(sizes[0] + [latent_dim], alpha, iterations, batch_size)
        self.decoder = Network([latent_dim] + sizes[1], alpha, iterations, batch_size)

        for i in range(len(self.encoder.weights)):
            self.encoder.weights[i] = np.abs(self.encoder.weights[i])

        for j in range(len(self.decoder.weights)):
            self.decoder.weights[j] = np.abs(self.decoder.weights[j])

        self.batch_size = batch_size
        self.iterations = iterations
        self.alpha = alpha

    def _forward(self, X):
        '''
        Propagates the input through the encoder, samples from the latent space
        and sends the sample through the decoder
        
        :param X: Model input
        '''
        # get the encoded/learned parameters from the last encoder layer
        latent = self.encoder._feedforward(X)
        self.mu = latent[0,:]
        self.sigma = np.exp(latent[1,:])

        # sample from the latent distribution with the learned parameters
        # epsilon for the reparameterization trick
        epsilon = np.random.normal(0, 1, size=(X.shape[1], self.latent_dim))
        z_sample = self.mu[:,None] + np.sqrt(self.sigma)[:,None] * epsilon
        
        # propagate the transposed (input as row vector/matrix with 1 column) sample trough the decoder
        return self.decoder._feedforward(z_sample.T)

    def _kl_divergence_loss(self):
        '''
        Calculates the Kullback-Leibler-Divergence
        
        '''
        d_s2 = 1 - 1 / (2 * (self.sigma + 1e-6))
        return np.vstack((self.mu, d_s2)).T

    def _backward(self, X, X_hat):
        '''
        Backpropagation for the encoder and the decoder

        :param X: Label
        :param X_hat: Model prediction
        '''
        n = len(self.decoder.weights)
        delta = -1 * self.decoder._loss(X, X_hat)[1] * self.decoder._activation(self.decoder._z[n-1])[1]
        decoder_weights = { n-1: delta @ self.decoder._a[n-1].T }
        decoder_biases = { n-1: delta }
        
        for i in reversed(range(n-1)):
            delta = self.decoder.weights[i+1].T @ delta * self.decoder._activation(self.decoder._z[i])[1]
            decoder_weights[i] = delta @ self.decoder._a[i].T
            decoder_biases[i] = delta

        m = len(self.encoder.weights)
        kl_loss = self._kl_divergence_loss()
        kl_delta = kl_loss.T * self.encoder._activation(self.encoder._z[m-1])[1]

        delta = self.decoder.weights[0].T @ delta * self.encoder._activation(self.encoder._z[m-1])[1]
        delta = delta + kl_delta
        encoder_weights = { m-1: delta @ self.encoder._a[m-1].T }
        encoder_biases = { m-1: delta }

        for i in reversed(range(m-1)):
            delta = self.encoder.weights[i+1].T @ delta * self.encoder._activation(self.encoder._z[i])[1]
            encoder_weights[i] = delta @ self.encoder._a[i].T
            encoder_biases[i] = delta

        return encoder_weights, encoder_biases, decoder_weights, decoder_biases

    def learn(self, X):
        '''
        Model training with stochastic gradient descent
        
        :param X: Input
        '''
        X_batch = X

        for _ in range(self.iterations):
            if self.batch_size > 0 and self.batch_size < X.shape[1]:
                k = np.random.choice(range(X.shape[1]), self.batch_size, replace=False)
                X_batch = X[k,:]
            
            X_hat = self._forward(X_batch)
            grad_encoder_weights, grad_encoder_biases, grad_decoder_weights, grad_decoder_biases = self._backward(X_batch, X_hat)

            for j in range(len(self.encoder.weights)):
                self.encoder.weights[j] -= self.alpha * grad_encoder_weights[j]
                self.encoder.biases[j] -= self.alpha * grad_encoder_biases[j].T

            for k in range(len(self.decoder.weights)):
                self.decoder.weights[k] -= self.alpha * grad_decoder_weights[k]
                self.decoder.biases[k] -= self.alpha * grad_decoder_biases[k].T

    def generate(self, z = None):
        '''
        Generates a new output

        '''
        if not np.any(z):
            z = np.random.normal(0, 1, size=(1, self.latent_dim))
        return self.decoder.predict(z)

    def encode_decode(self, X):
        '''
        Propagates the input through the VAE

        :param X: Input as row vector or a matrix with 1 column
        '''
        return self._forward(X)