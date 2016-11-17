
import numpy as np
import os

import theano
import theano.tensor as T
import lasagne



'''
Image arrays have the shape (N, 3, 32, 32), where N is the size of the
corresponding set. This is the format used by Lasagne/Theano. To visualize the
images, you need to change the axis order, which can be done by calling
np.rollaxis(image_array[n, :, :, :], 0, start=3).

Each image has an associated 40-dimensional attribute vector. The names of the
attributes are stored in self.attr_names.
'''

""" Hyperparameters Used:

1. learning rate.
2. dropout value.
3.....

"""

data_path = "/home/lmb/Celeb_data"


class Network:

    def load_data(self):
        self.train_images = np.float32(np.load(os.path.join(
                data_path, "train_images_32.npy"))) / 255.0
        self.train_labels = np.uint8(np.load(os.path.join(
                data_path, "train_labels_32.npy")))
        self.val_images = np.float32(np.load(os.path.join(
                data_path, "val_images_32.npy"))) / 255.0
        self.val_labels = np.uint8(np.load(os.path.join(
                data_path, "val_labels_32.npy")))
        self.test_images = np.float32(np.load(os.path.join(
                data_path, "test_images_32.npy"))) / 255.0
        self.test_labels = np.uint8(np.load(os.path.join(
                data_path, "test_labels_32.npy")))

        with open(os.path.join(data_path, "attr_names.txt")) as f:
            self.attr_names = f.readlines()[0].split()

    
    
    def build_Network(self, inputVar = None, depth = None, num_units = None dropOne = None, dropSecond=None):
        
        inputLayer = lasagne.layers.input(shape=(None, 3, 32, 32), inputVar = inputVar)
        
        """ Applying dropout to the output of the input layer data"""
        
        if dropOne:
            inputDropLayer = lasagne.layers.DropoutLayer(inputLayer, p = dropOne)
        
        for _ in range(depth):
            layerFC = lasagne.layers.DenseLayer(inputDropLayer, num_units, nonlinearity =       
                                                        lasagne.nonlinearities.rectify='relu')
            if dropSecond:
                layerFC = lasagne.layers.DropoutLayer(layerFC, p = dropSecond)
                
        """ Adding first 2D convolution layer to the network"""
            
        convHiddenLayerOut1 = lasagne.layers.Conv2DLayer(inputDropLayer,  num_filters=32, filter_size=(5, 5),
                                                        nonlinearity=lasagne.nonlinearities.rectify,
                                                        W=lasagne.init.HeNormal(gain='relu'))
        
        """ First Max-pooling layer of size 2 in both dimensions """
        
        poolLayer1 = lasagne.layers.MaxPool2DLayer(convHiddenLayerOut1, pool_size=(2, 2))
        
        
         """ Adding Second 2D convolution layer to the network"""
            
        convHiddenLayerOut2 = lasagne.layers.Conv2DLayer(poolLayer1,  num_filters=32, filter_size=(5, 5),
                                                        nonlinearity=lasagne.nonlinearities.rectify,
                                                        W=lasagne.init.HeNormal(gain='relu'))
        
        """ Second Max-pooling layer of size 2 in both dimensions """
        
        poolLayer2 = lasagne.layers.MaxPool2DLayer(convHiddenLayerOut2, pool_size=(2, 2))
        
        
        """ Output layer using the softmax."""
        
        softmax = lasagne.nonlinearities.softmax
        
        network = lasagne.layers.DenseLayer(lasagne, 2, nonlinearities=softmax)
        
        return Network
    
    
    def iterate_minibatches(self, inputs, targets, batchsize, shuffle):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
    
    
    
    def train(self, network, trainY, valX, valY, inputVar, learning_rate, nr_epoch = 500, batchSize = 500,              descent_type):
         
        print ("Training...")
        
        """ creating two theano variables for input and output"""
        
        #inputVar = T.tensor4('inputs')
        groundTruthVal = T.ivector('targets')
        
        prediction = lasagne.layers.get_output(network)
        
        train_loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        
        train_loss = train_loss.mean()
    
        """ Add lasagne.regularization here """
        """ Updating parameters at each training step and  getting training loss"""

       params = lasagne.layers.get_all_parameters(network, trainable=True)
       
       if descent_type == 'sgd':
           updates = lasagne.update.sgd(loss, params, learning_rate)
       else if descent_type == 'adam':
           updates = lasagne.update.adam(loss, params, learning_rate)
           
           
       """ get test and validation loss """
       
       test_prediction = lasagne.layers.get_output(network, deterministic=True)
       test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
       test_loss = test_loss.mean()
       
       "A compile function from theano for update and loss"
       
       train_func = theano.function([inputVar, groundTruthVal], train_loss, updates=updates)
       
       "Compile a second function to compute validation loss."
       
       val_function = theano.function([input_var, target_var], test_loss)
        
       
        
        
        
nn = Network()    
nn.load_data()


