package com.assignment.mlp;

import java.io.File;
import java.util.Scanner;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class App {
    public static void main(String[] args) {
        try (Scanner scanner = new Scanner(System.in)) {
            System.out.println("Choose mode:");
            System.out.println("1. Train a new model");
            System.out.println("2. Make predictions with existing model");
            System.out.println("3. Exit");
            System.out.print("Enter your choice (1, 2 or 3): ");
            
            
            int choice = scanner.nextInt();
            scanner.nextLine(); // consume newline
            switch (choice) {
                case 1 -> {
                    System.out.println("Choose Training mode:");
                    System.out.println("1. Train a single model");
                    System.out.println("2. Run full grid search");
                    System.out.println("3. Exit");
                    System.out.print("Enter your choice (1, 2 or 3): ");
                    choice = scanner.nextInt();
                    scanner.nextLine();
                    switch (choice) {
                        case 1 -> trainSingleModel(scanner);
                        case 2 -> runGridSearch(scanner);
                        case 3 -> System.out.println("Exiting the application.");
                        default -> System.err.println("Invalid choice. Please enter 1, 2 or 3.");
                    }
                }
                case 2 -> predictMode(scanner);
                case 3 -> System.out.println("Exiting the application.");
                default -> System.err.println("Invalid choice. Please enter 1, 2 or 3.");
            }
            
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
    
    private static void trainSingleModel(Scanner scanner) throws Exception {
        // Your existing trainMode implementation
        System.out.println("\n=== Single Model Training ===");
        UserInput input = DataLoader.getUserInput(scanner);
        String dataPath = input.getDataPath();
        long seed = input.getSeed();
        int batchSize = input.getBatchSize();
        double learningRate = input.getLearningRate();
        int numInputs = 5;
        int epochs = 50;
        
        System.out.println("\nStarting MLP with:");
        System.out.println("Data path: " + dataPath);
        System.out.println("Random seed: " + seed);
        System.out.println("Batch size: " + batchSize);
        System.out.println("Learning rate: " + learningRate);
        System.out.println("Epochs: " + epochs);
        
        DataSetIterator trainData = DataLoader.loadData(dataPath, batchSize);
        MultiLayerNetwork model = MultilayerPerceptron.buildNetwork(seed, numInputs, learningRate);
        
        long startTime = System.currentTimeMillis();
        for (int i = 0; i < epochs; i++) {
            model.fit(trainData);
            trainData.reset();
        }
        long trainingTime = (System.currentTimeMillis() - startTime) / 1000;
        
        System.out.println("\nTraining completed in " + trainingTime + " seconds.");
        Evaluation eval = model.evaluate(trainData);
        System.out.println(eval.stats());
        
        System.out.print("\nDo you want to save the trained model? (y/n): ");
        String saveChoice = scanner.nextLine().trim().toLowerCase();
        if (saveChoice.equals("y") || saveChoice.equals("yes")) {
            System.out.print("Enter filename to save model (e.g., myModel.zip): ");
            String modelFilename = scanner.nextLine().trim();
            File modelFile = new File(modelFilename);
            model.save(modelFile, true);
            System.out.println("Model saved to: " + modelFile.getAbsolutePath());
        }
    }
    
    private static void runGridSearch(Scanner scanner) throws Exception {
        System.out.println("\n=== Grid Search Mode ===");
        System.out.println("WARNING: This will test all hyperparameter combinations and may take a long time!");
        System.out.print("Enter path to training data CSV file: ");
        String dataPath = scanner.nextLine().trim();
        
        System.out.print("Enter base random seed (default 527): ");
        long baseSeed = scanner.hasNextLong() ? scanner.nextLong() : 527;
        scanner.nextLine(); // consume newline
        
        System.out.println("Starting grid search...");
        
        GridSearchExecutor executor = new GridSearchExecutor(dataPath, 5, baseSeed);
        executor.executeFullGridSearch();
        
        System.out.println("Grid search completed! Results saved to grid_search_results.csv");
    }

    private static void predictMode(Scanner scanner) throws Exception {
        System.out.println("\n=== Prediction Mode ===");
        // Load saved model
        System.out.print("Enter path to saved model file: ");
        String modelPath = scanner.nextLine().trim();
        File modelFile = new File(modelPath);
        
        if (!modelFile.exists()) {
            throw new IllegalArgumentException("Model file not found: " + modelFile.getAbsolutePath());
        }
        
        MultiLayerNetwork model = MultiLayerNetwork.load(modelFile, true);
        
        // Get data for prediction
        System.out.print("Enter path to CSV file for predictions: ");
        String dataPath = scanner.nextLine().trim();
        int batchSize = 32;
        
        DataSetIterator predictionData = DataLoader.loadData(dataPath, batchSize);
        
        // Make predictions
        INDArray predictions = model.output(predictionData);
        
        // Display predictions
        System.out.println("\nPredictions:");
        for (int i = 0; i < predictions.length(); i++) {
            double prob = predictions.getDouble(i);
            int predictedClass = prob > 0.5 ? 1 : 0;
            System.out.printf("Sample %d: Class %d (Probability: %.4f)%n", 
                            i, predictedClass, prob);
        }
    }
}