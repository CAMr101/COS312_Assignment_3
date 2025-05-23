package com.assignment.mlp;

public class UserInput {
    private String dataPath;
    private long seed;
    private int batchSize;
    private double learningRate;

    // Constructor
    public UserInput(String dataPath, long seed, int batchSize, double learningRate) {
        this.dataPath = dataPath;
        this.seed = seed;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
    }

    // Getters
    public String getDataPath() {
        return dataPath;
    }

    public long getSeed() {
        return seed;
    }

    // Setters (optional)
    public void setDataPath(String dataPath) {
        this.dataPath = dataPath;
    }

    public void setSeed(long seed) {
        this.seed = seed;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    
}
