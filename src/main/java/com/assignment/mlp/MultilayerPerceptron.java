package com.assignment.mlp;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class MultilayerPerceptron {
    public static MultiLayerNetwork buildNetwork(long seed, int numInputs, double learningRate, int[] neuronsPerLayer, Activation activation, WeightInit weightInit) {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(weightInit)
                .updater(new Adam(learningRate))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(neuronsPerLayer[0]) // Number of neurons in the first hidden layer
                        .activation(activation) 
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(neuronsPerLayer[0])
                        .nOut(neuronsPerLayer[1]) // Number of neurons in the second hidden layer
                        .activation(activation)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(neuronsPerLayer[1])
                        .nOut(neuronsPerLayer[2]) // Number of neurons in the third hidden layer
                        .activation(activation) 
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT) // Binary Cross-Entropy
                        .nIn(neuronsPerLayer[2])
                        .nOut(1) // Number of neurons in the output layer
                        .activation(Activation.SIGMOID) // Use sigmoid for binary classification	
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        return model;   
    }
}
