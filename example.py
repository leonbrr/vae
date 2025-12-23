from src.network import Network
from src.variational_autoencoder import VAE
from src.image import image
from matplotlib import pyplot as plt
import numpy as np

def encoding():
    '''
    Example of an encoder network by using the pixel matrix from a number of the MNIST dataset
    '''
    # plt.matshow(image, cmap='gray', clim=(0,1))
    # plt.show()

    # define the network and the input
    ex = Network(sizes=[784, 200, 784], alpha=0.1, iterations=5, batch_size=0)
    im_temp = image.reshape((784, 1))

    # prediction before training
    pred = ex.predict(im_temp)
    pred = pred.reshape((28, 28))

    # prediction after training
    ex.train(im_temp, im_temp)
    trained = ex.predict(im_temp)
    trained = trained.reshape((28, 28))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    ax[0].matshow(image, cmap='gray', clim=(0,1))
    ax[0].title.set_text('Original input')
    ax[1].matshow(pred, cmap='gray', clim=(0,1))
    ax[1].title.set_text('Untrained network')
    ax[2].matshow(trained, cmap='gray', clim=(0,1))
    ax[2].title.set_text('Trained encoder output')
    fig.suptitle('Image reconstruction with encoder network')
    plt.show()

def use_vae():
    '''
    Example usage of the VAE class by using the pixel matrix from a number of the MNIST dataset
    '''

    # latent dim is 2 because two parameters have to be estimated per KL-Divergence
    vae = VAE(sizes=[[784, 200], [200, 784]], latent_dim=2, alpha=0.1, iterations=10, batch_size=0)
    im_temp = image.reshape((784, 1))
    vae.learn(im_temp)

    # generate first image
    out = vae.encode_decode(im_temp)
    out = out.reshape((28, 28))

    # generate second image
    out2 = vae.encode_decode(im_temp)
    out2 = out2.reshape((28, 28))

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    ax[0].matshow(image, cmap='gray', clim=(0,1))
    ax[0].title.set_text('Original input')
    ax[1].matshow(out, cmap='gray', clim=(0,1))
    ax[1].title.set_text('First image')
    ax[2].matshow(out2, cmap='gray', clim=(0,1))
    ax[2].title.set_text('Second image')
    fig.suptitle('Generation process of two images with the VAE')
    plt.show()


#ex = Network(sizes=[2,2,1], alpha=1, iterations=1, batch_size=0)
#ex.print()
#a = np.array([[0.35, 0.7]]).T
#b = ex.predict(a)
#print(b, 'prediction')
#ex.train(a,0.5)
#print(ex.predict(a), 'after training')

#vae = VAE(sizes=[[5, 3], [3, 5]], latent_dim=2, alpha=0.1, iterations=1, batch_size=0)
#out = vae.encode_decode(np.array([[0.2, 0.3, 0.4, 0.5, 0.6]]).T)
#print(out, 'out')
#vae.learn(np.array([[0.2, 0.3, 0.4, 0.5, 0.6]]).T)


if __name__ == "__main__":
    encoding()
    use_vae()