package com.assignment.mlp;

import org.nd4j.linalg.activations.Activation; 
import org.deeplearning4j.nn.weights.WeightInit;


public class HyperparameterConfig {
    private double learningRate;
    private int batchSize;
    private int epochs;
    private int layer1Neurons;
    private int layer2Neurons;
    private int layer3Neurons;
    private Activation activation;
    private WeightInit weightInit;
    
    // Constructor, getters, and setters
    public HyperparameterConfig(double learningRate, int batchSize, int epochs,
                              int layer1Neurons, int layer2Neurons, int layer3Neurons,
                              Activation activation, WeightInit weightInit) {
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.epochs = epochs;
        this.layer1Neurons = layer1Neurons;
        this.layer2Neurons = layer2Neurons;
        this.layer3Neurons = layer3Neurons;
        this.activation = activation;
        this.weightInit = weightInit;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getEpochs() {
        return epochs;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public int getLayer1Neurons() {
        return layer1Neurons;
    }

    public void setLayer1Neurons(int layer1Neurons) {
        this.layer1Neurons = layer1Neurons;
    }

    public int getLayer2Neurons() {
        return layer2Neurons;
    }

    public void setLayer2Neurons(int layer2Neurons) {
        this.layer2Neurons = layer2Neurons;
    }

    public int getLayer3Neurons() {
        return layer3Neurons;
    }

    public void setLayer3Neurons(int layer3Neurons) {
        this.layer3Neurons = layer3Neurons;
    }

    public Activation getActivation() {
        return activation;
    }

    public void setActivation(Activation activation) {
        this.activation = activation;
    }

    public WeightInit getWeightInit() {
        return weightInit;
    }

    public void setWeightInit(WeightInit weightInit) {
        this.weightInit = weightInit;
    }


    
}