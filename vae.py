import numpy as np
import time
import os
import sys
import theano 
import theano.tensor as T
import lasagne
import theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from scipy.stats import norm


def load_dataset():
    
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print "Downloading... %s", %filename
        urlretrieve(source + filename, filename)
    
    import gzip
    
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        
        data = data.reshape(-1, 1, 28, 28).np.transpose(0, 1, 3, 2)
        
        return data
    
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    X_train,  X_val = X_train([:-1000], [-1000:])
    
    return X_train, X_test, X_val


def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
    
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
    
        yield inputs[excerpt]
        
class GaussianSmapleLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng = None, **kwargs):
        self.rng = rng if rng else RandomStreams(lasagne.random.get_rng().randint(1,2147462579))
        super(GaussianSmapleLayer, self).__init__([mu, logsigma], **kwargs)
            
            return input_shapes[0]
     
        def get_output_for(self, inputs, withnoise = True, **kwargs):
            mu, logsigma = inputs
            
            shape = (self.input_shapes[0][0] or inputs[0].shapes[0]
                    self.input_shapes[0][1] or inputs[0].shapes[1])
            
            if withnoise:
                return mu + logsigma * self.rng.normal(shape)
        
            return mu, logsigma
        
            
                
def build_VAE(input_var, L=2, binary=True):
    
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var = input_var)
    network = lasagne.layers.DenseLayer(network, num_units = 1024 = nn.nonlinearities.tanh)  
    network = lasagne.layers.DenseLayer(network, num_units = 2, nonlinearity = None)
    network = GaussianSampleLayer(l_enc_mu, l_enc_logsigma, name='Z')
    
    network = lasagne.layers.DenseLayer(l_dec_hid, num_units = x_dim,
                    nonlinearity = nn.nonlinearities.sigmoid,
                    W = nn.init.GlorotUniform(),
                    b = nn.init.Constant(0.))
    
    network = l_dec_logsigma = nn.layers.DenseLayer(network, num_units=x_dim,
                    W = nn.init.GlorotUniform() if W_dec_ls is None else W_dec_ls,
                    b = nn.init.Constant(0) if b_dec_ls is None else b_dec_ls,
                    nonlinearity = lambda a: T.nnet.relu(a+relu_shift)-relu_shift)
    networkk = GaussianSampleLayer(network, l_dec_logsigma, name='dec_output')           
    
    train_fn = theano.function([input_var], loss, updates=updates)
    val_fn = theano.function([input_var], test_loss)

    print("Starting training...")
    batch_size = 100
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, batch_size, shuffle=True):
            this_err = train_fn(batch)
            train_err += this_err
            train_batches += 1
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, batch_size, shuffle=False):
            err = val_fn(batch)
            val_err += err
            val_batches += 1
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    test_err = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, batch_size, shuffle=False):
        err = val_fn(batch)
        test_err += err
        test_batches += 1
    test_err /= test_batches
    print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err))                            


def main(L=2, z_dim=2, n_hid=1024, num_epochs=300, binary=True):
    print("Loading data...")
    X_train, X_val, X_test = load_dataset()
    width, height = X_train.shape[2], X_train.shape[3]
    input_var = T.tensor4('inputs')
    
    
    
