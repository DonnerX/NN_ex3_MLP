#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron
from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot
import numpy as np

# class test():
#     def __init__(self):
#         self.l=5
#
#     def change(self):
#         self.l=8


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=False)

    myMLP = MultilayerPerceptron(data.trainingSet,
                                 data.validationSet,
                                 data.testSet,
                                 learningRate=0.005,
                                 epochs=30)
    # myMLP.fortest()
    myMLP.train()

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    MLPPred = myMLP.evaluate()
    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("\nResult of the MLP:")
    evaluator.printAccuracy(data.testSet, MLPPred)

    # Draw
    plot = PerformancePlot("MLP validation")
    plot.draw_performance_epoch(myMLP.performances, myMLP.epochs)



if __name__ == '__main__':
    main()