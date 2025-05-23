package com.assignment.mlp;

import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;

public class UserInput {
    private String dataPath;
    private long seed;
    private double learningRate;
    private int batchSize;
    private int epochs;
    private int[] neuronsPerLayer;
    private Activation activation;
    private WeightInit weightInit;

    // Constructor
    public UserInput(String dataPath, long seed, int batchSize, double learningRate) {
        this.dataPath = dataPath;
        this.seed = seed;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
    }

    public UserInput(String dataPath, long seed, int batchSize, double learningRate, int epochs,
            int[] neuronsPerLayer, Activation activation, WeightInit weightInit) {
        this.dataPath = dataPath;
        this.seed = seed;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.neuronsPerLayer = neuronsPerLayer;
        this.activation = activation;
        this.weightInit = weightInit;
    }

    public String getDataPath() {
        return dataPath;
    }

    public void setDataPath(String dataPath) {
        this.dataPath = dataPath;
    }

    public long getSeed() {
        return seed;
    }

    public void setSeed(long seed) {
        this.seed = seed;
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

    public int[] getNeuronsPerLayer() {
        return neuronsPerLayer;
    }

    public void setNeuronsPerLayer(int[] neuronsPerLayer) {
        this.neuronsPerLayer = neuronsPerLayer;
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


