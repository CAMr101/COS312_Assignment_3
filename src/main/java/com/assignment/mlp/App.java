package com.assignment.mlp;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;
import java.util.Scanner;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class App {
    public static void main(String[] args) {
        final String OS_String = System.getProperty("os.name");

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                UtilityFunctions.clearConsole(OS_String);
                System.out.println("=== MLP Application ===");
                System.out.println("1. Train a new model");
                System.out.println("2. Make predictions with existing model");
                System.out.println("3. Exit");
                System.out.print("Enter your choice (1, 2, or 3): ");

                int choice = -1;
                if (scanner.hasNextInt()) {
                    choice = scanner.nextInt();
                }
                scanner.nextLine(); // consume newline

                switch (choice) {
                    case 1 -> {
                        UtilityFunctions.clearConsole(OS_String);
                        System.out.println("=== Training Mode ===");
                        System.out.println("1. Train a single model");
                        System.out.println("2. Run full grid search");
                        System.out.println("3. Back to Main Menu");
                        System.out.print("Enter your choice (1, 2, or 3): ");
                        int trainingChoice = -1;
                        if (scanner.hasNextInt()) {
                            trainingChoice = scanner.nextInt();
                        }
                        scanner.nextLine();

                        switch (trainingChoice) {
                            case 1 -> trainSingleModel(scanner, OS_String);
                            case 2 -> runGridSearch(scanner, OS_String);
                            case 3 -> System.out.println("Returning to main menu.");
                            default -> System.err.println("Invalid choice. Please enter 1, 2, or 3.");
                        }
                        System.out.println("\nPress Enter to continue...");
                        scanner.nextLine(); // Wait for user to read output
                    }
                    case 2 -> {
                        predictMode(scanner, OS_String);
                        System.out.println("\nPress Enter to continue...");
                        scanner.nextLine(); 
                    }
                    case 3 -> {
                        System.out.println("Exiting the application. Goodbye!");
                        return;
                    }
                    default -> {
                        System.err.println("Invalid choice. Please enter 1, 2, or 3.");
                        System.out.println("Press Enter to continue...");
                        scanner.nextLine();
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("An unexpected error occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void trainSingleModel(Scanner scanner, String OS_String) throws Exception {
        UtilityFunctions.clearConsole(OS_String);
        System.out.println("=== Single Model Training ===\n"); 
        UserInput input = DataLoader.getUserInput(scanner);
        String dataPath = input.getDataPath();
        long seed = input.getSeed();
        int batchSize = input.getBatchSize();
        double learningRate = input.getLearningRate();
        int epochs = input.getEpochs();
        int[] neuronsPerLayer = input.getNeuronsPerLayer();
        Activation activation = input.getActivation();
        WeightInit weightInit = input.getWeightInit();

        System.out.println("\nStarting MLP with the following parameters:");
        System.out.println("  Data path: " + dataPath);
        System.out.println("  Random seed: " + seed);
        System.out.println("  Batch size: " + batchSize);
        System.out.println("  Learning rate: " + learningRate);
        System.out.println("  Epochs: " + epochs);
        System.out.printf("  Activation Function: %s%n", activation);
        System.out.printf("  Weight Initialization: %s%n", weightInit); 

        System.out.print("  Hidden layer neurons: [");
        for (int i = 0; i < neuronsPerLayer.length; i++) {
            System.out.print(neuronsPerLayer[i]);
            if (i < neuronsPerLayer.length - 1) {
                System.out.print(", ");
            }
        }
        System.out.println("]\n");

        DataSetIterator trainData = DataLoader.loadData(dataPath, batchSize);
        MultiLayerNetwork model = MultilayerPerceptron.buildNetwork(seed, 5, learningRate, neuronsPerLayer, activation, weightInit);

        long startTime = System.currentTimeMillis();
        for (int i = 0; i < epochs; i++) {
            model.fit(trainData);
            trainData.reset();
            System.out.printf("\rEpoch %d/%d completed.%n", (i + 1), epochs);
        }
        long trainingTime = (System.currentTimeMillis() - startTime) / 1000;

        System.out.println("\nTraining completed in " + trainingTime + " seconds.");
        trainData.reset(); // Reset iterator for evaluation
        Evaluation eval = model.evaluate(trainData);
        System.out.println("\n=== Model Evaluation ===");
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

    private static void runGridSearch(Scanner scanner, String OS_String) throws Exception {
        UtilityFunctions.clearConsole(OS_String);
        System.out.println("=== Grid Search Mode ===\n");
        System.out.println("WARNING: This will test all hyperparameter combinations and may take a long time!");
        System.out.print("Enter path to training data CSV file: ");
        String dataPath = scanner.nextLine().trim();

        System.out.print("Enter base random seed (default 527): ");
        long baseSeed = scanner.hasNextLong() ? scanner.nextLong() : 527;
        scanner.nextLine();

        System.out.println("\nInitiating grid search...");

        GridSearchExecutor executor = new GridSearchExecutor(dataPath, 5, baseSeed);
        executor.executeFullGridSearch();

        System.out.println("\nGrid search process concluded.");
    }

    private static void predictMode(Scanner scanner, String OS_String) throws Exception {
        UtilityFunctions.clearConsole(OS_String);
        System.out.println("=== Prediction Mode ===\n");

        // Load saved model
        System.out.print("Enter path to saved model file (e.g., myModel.zip): ");
        String modelPath = scanner.nextLine().trim();
        File modelFile = new File(modelPath);

        if (!modelFile.exists()) {
            throw new IllegalArgumentException("Model file not found: " + modelFile.getAbsolutePath());
        }
        // Load the model
        MultiLayerNetwork model = MultiLayerNetwork.load(modelFile, true);
        if (model == null) {
            throw new IOException("Failed to load model from file: " + modelFile.getAbsolutePath());
        }

        UtilityFunctions.clearConsole(OS_String);
        System.out.println("Model loaded successfully: " + modelFile.getName());

        // Get data for prediction
        System.out.print("Enter path to CSV file for predictions: ");
        String dataPath = scanner.nextLine().trim();
        int batchSize = 32; // Default batch size for prediction

        DataSetIterator predictionDataIterator = DataLoader.loadData(dataPath, batchSize);

        // Get output file path from user
        System.out.print("Enter the path for the output CSV file (e.g., predictions.csv): ");
        String outputCsvPath = scanner.nextLine().trim();

        // Make predictions and write to CSV
        System.out.println("\nGenerating predictions and writing to CSV...");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputCsvPath))) {
            // CSV header
            writer.write("Sample_ID,Predicted_Class,Probability\n");

            int sampleIndex = 0;
            // Iterate through the DataSetIterator to process data in batches
            while (predictionDataIterator.hasNext()) {
                INDArray features = predictionDataIterator.next().getFeatures();
                INDArray batchPredictions = model.output(features);

                for (int i = 0; i < batchPredictions.rows(); i++) {
                    double prob = batchPredictions.getDouble(i);
                    int predictedClass = prob > 0.5 ? 1 : 0;
                    sampleIndex++;

                    // Write to console
                    System.out.printf("Sample %d: Predicted Class %d (Probability: %.4f)%n",
                            sampleIndex, predictedClass, prob);

                    // Write to CSV
                    writer.write(String.format(Locale.ROOT, "%d,%d,%.4f%n", sampleIndex, predictedClass, prob));
                }
            }
            System.out.println("\nPredictions successfully written to: " + outputCsvPath);

        } catch (IOException e) {
            System.err.println("Error writing predictions to CSV file: " + e.getMessage());
            throw e;
        }

        System.out.println("\nPrediction process complete.");
    }
}
