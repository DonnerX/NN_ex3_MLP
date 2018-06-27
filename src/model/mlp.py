
import numpy as np
from util.loss_functions  import CrossEntropyError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

import sys

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='ce', learningRate=0.01, epochs=50, l2_para=0):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        self.error = 0
        self.l2_para = l2_para

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        if loss == 'ce':
            self.loss = CrossEntropyError()
        elif loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(train.input.shape[1], 128, 
                           None, inputActivation, False))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(128, 10, 
                           None, outputActivation, True))

        self.inputWeights = inputWeights

        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """

        for layer in self.layers:
            inp = np.insert(layer.forward(inp),0,1,axis=0)
        outp = self.layers[-1].outp
        return outp

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        self.error += self.loss.calculateError(target,self.layers[-1].outp)
    
    def _update_weights(self, learningRate, l2_para):
        """
        Update the weights of the layers by propagating back the error
        """
        for i in range(0, len(self.layers)):
            self.layers[i].updateWeights(learningRate, l2_para)
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self.error = 0
            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")
            if self.error == 0:
                print("it's full trained")
                return

    def _train_one_epoch(self):
        for img, label in zip(self.trainingSet.input,
                              self.trainingSet.label):
            temp_l = np.zeros(10)
            temp_l[label] = 1   #transform Decimal label to Binary
            self._feed_forward(img)
            self._compute_error(temp_l)
            next_weight = 1.0
            next_derivatives = self.loss.calculateDerivative(temp_l, self.layers[-1].outp)
            for i in range(len(self.layers)-1,-1,-1):
                next_derivatives = self.layers[i].computeDerivative(next_derivatives, next_weight) # for next iter
                next_weight = self.layers[i].weights[1:] # for next iter , get the weight without bias
                #self.layers[i].updateWeights(self.learningRate)
            self._update_weights(self.learningRate, self.l2_para)



    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        outp = self._feed_forward(test_instance)
        result = np.where(outp == np.max(outp))
        return result[0]
        

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
