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
    public static MultiLayerNetwork buildNetwork(long seed, int numInputs, double learningRate) {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(64) // Number of neurons in the first hidden layer
                        .activation(Activation.RELU) // Use ReLU activation function
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(64)
                        .nOut(32) // Number of neurons in the second hidden layer
                        .activation(Activation. SWISH) // Use Swish activation function
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(32)
                        .nOut(16) // Number of neurons in the third hidden layer
                        .activation(Activation.RELU) 
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT) // Binary Cross-Entropy
                        .nIn(16)
                        .nOut(1) // Number of neurons in the output layer
                        .activation(Activation.SIGMOID) // Use sigmoid for binary classification	
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        return model;   
    }
}
