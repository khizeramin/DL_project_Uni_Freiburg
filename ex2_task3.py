
import matplotlib.pyplot as plt
import numpy as np
import os

import time
import theano
import theano.tensor as T
import lasagne

from lasagne.layers import get_output
from lasagne.layers import get_output_shape
 

# '''
# Image arrays have the shape (N, 3, 32, 32), where N is the size of the
# corresponding set. This is the format used by Lasagne/Theano. To visualize the
# images, you need to change the axis order, which can be done by calling
# np.rollaxis(image_array[n, :, :, :], 0, start=3).

# Each image has an associated 40-dimensional attribute vector. The names of the
# attributes are stored in self.attr_names.
# '''

# """ Hyperparameters Used:

# 1. learning rate.
# 2. dropout value.
# 3.....

# """

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

        print ("Done.")

    def build_network(self, inputVar=None):
        print(type(inputVar))
        print "..."
        #creating an input layer layer for the network
        network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=inputVar)
        #creating a 2D convolution layer with filter size=5x5 
        network = lasagne.layers.Conv2DLayer(
                        network, num_filters=10, filter_size=(5, 5),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        W=lasagne.init.HeNormal(gain='relu'))

        
        #network = lasagne.layers.Conv2DLayer(network, num_filters=10, filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal=(gain='relu'))
        #creating a maxpool layer 
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        #Another Conv layer is added to the network
        network = lasagne.layers.Conv2DLayer(
                        network, num_filters=10, filter_size=(5, 5),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        W=lasagne.init.HeNormal(gain='relu'))
        #network = lasagne.layers.Conv2DLayer(network, num_filters=10, filter_size=(5, 5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.HeNormal=(gain='relu'))
        #Maxpool layer 
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        #Two Fully connected layers are added here.
        network = lasagne.layers.DenseLayer(
                        lasagne.layers.dropout(network, p=.5),
                        num_units=256,
                        nonlinearity=lasagne.nonlinearities.rectify,
                        W=lasagne.init.HeNormal(gain='relu'))

        network = lasagne.layers.DenseLayer(
                        lasagne.layers.dropout(network, p=.5),
                        num_units=40, nonlinearity=lasagne.nonlinearities.sigmoid,
                        W=lasagne.init.HeNormal(gain='relu' ))

        return network

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
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
                                                                

    def train(self, trainX, trainY, valX, valY):

        print (" Started Training... ")

        descent_type="sgd"
        nr_epochs =10 
        learning_rate = 0.01
        batch_size=500

        input_var = T.tensor4('inputs')

        target_var = T.imatrix('targets')

        network = self.build_network(input_var)
        

        predictions = lasagne.layers.get_output(network, determinitic = False)
        loss = lasagne.objectives.binary_crossentropy(predictions, target_var)
        loss = loss.mean()
           
        val_predictions = lasagne.layers.get_output(network, deterministic=True)
        val_loss = lasagne.objectives.binary_crossentropy(val_predictions, target_var)
        val_loss = val_loss.mean()
        
        #creating update expressions for training.i.e. how to modify the paramteres at each step
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.sgd(loss, params, learning_rate=0.01)
        print ("netrwork returned well")

        train_fn = theano.function([input_var, target_var], loss, updates=updates)
        
        #Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], val_loss)
        
        for epoch in range(nr_epochs):
            train_err= 0
            train_batches=0
            start_time = time.time()
            for batches in self.iterate_minibatches(trainX, trainY, 10, shuffle=True):
                inputs, targets = batches
                train_err += train_fn(inputs, targets)
                train_batches +=1
                #print (train_err)
            val_err=0
            #val_acc=0
            val_batches=0
            for batches in self.iterate_minibatches(valX, valY, 10, shuffle=False):
                inputs, targets = batches
                err = val_fn(inputs, targets)
                val_err += err
                val_batches +=1
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, nr_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  val loss:\t\t{:.6f}".format(val_err / val_batches))
            #print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

    def display_filters(self, layer):
        
        W = layer.W.get_value()
        shape = W.shape 
        nrows = np.ceil(np.sqrt(shape[0])).astype(int)
        ncols = nrows
        

    def main(self):

        self.load_data()

        i =  self.attr_names.index("Male")
        
        trainX=self.train_images[:1000,:]
        valX=self.val_images[:1000,:]
        trainY=self.train_labels[:1000,:]
        valY=self.val_labels[:1000,:]
        
        #retunedModel = self.train(trainX, trainY, valX, valY)
        
        print ("cool")
        
#nn = Network()
#nn.main()

##################################### Task: 3 Display filters from the first Conv layer######################################


if __name__=="__main__":
    nn = Network()
    inputVar = T.tensor4('inputs')
    params = lasagne.layers.get_all_param_values(nn.build_network(inputVar))

    # Filters from the first conv layer
    filters_FC1 = params[0]
    
    #image = filters_FC1[0,0]
    fig = plt.figure()
    count = 1
        
    for i in range (10):
        for j in range(3):
            plt.subplot(3,10,count)
            plt.imshow(filters_FC1[i,j])
            count += 1

    plt.show()
    
       # AXES.FLATTEN()[I].IMSHOW(IMAGE)
    #plt.show()
    


    
  
    
    
    
