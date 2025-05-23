package com.assignment.mlp;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import java.io.File;
import java.nio.file.Paths;
import java.util.Scanner;

public class DataLoader {
    
    public static DataSetIterator loadData(String filePath, int batchSize) throws Exception {
        // Convert to absolute path if not already
        File file = new File(filePath);
        if (!file.isAbsolute()) {
            String currentDir = System.getProperty("user.dir");
            file = Paths.get(currentDir, filePath).toFile();
        }

        // Verify file exists
        if (!file.exists()) {
            throw new IllegalArgumentException("File not found: " + file.getAbsolutePath());
        }

        // Initialize CSV reader (skip 1 header line, comma delimiter)
        RecordReader recordReader = new CSVRecordReader(1, ',');
        recordReader.initialize(new FileSplit(file));

        // Create iterator for the dataset
        DataSetIterator iterator = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
            .classification(5, 1) // Column 5 is label, binary classification
            .build();

        // Fit the normalizer to the training data 
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(iterator); 

        // Reset the iterator and set the normalizer
        iterator.reset();
        iterator.setPreProcessor(normalizer);

        return iterator;
    }

    public static UserInput getUserInput(Scanner scanner) {
        Activation activation;
        WeightInit weightInit;

        System.out.println("Enter the path to the CSV file (relative or absolute):");
        String dataPath = scanner.nextLine().trim();
        
        System.out.println("Enter a random seed (default 527):");
        long seed = scanner.hasNextLong() ? scanner.nextLong() : 527;
        scanner.nextLine(); // consume newline

        System.out.println("Enter learning rate (default 0.0005):");
        double learningRate = scanner.hasNextDouble() ? scanner.nextDouble() : 0.0005;
        scanner.nextLine(); // consume newline

        System.out.println("Enter batch size (default 128):");
        int batchSize = scanner.hasNextInt() ? scanner.nextInt() : 128;
        scanner.nextLine(); // consume newline

        System.out.println("Enter number of epochs (default 150):");
        int epochs = scanner.hasNextInt() ? scanner.nextInt() : 150;
        scanner.nextLine(); // consume newline

        int[] neuronsPerLayer = new int[3];
        for (int i = 0; i < 3; i++) {
            System.out.printf("Enter number of neurons for hidden layer %d (default 64):", i + 1);
            neuronsPerLayer[i] = scanner.hasNextInt() ? scanner.nextInt() : 64;
            scanner.nextLine(); // consume newline
        }

        System.out.println("Enter activation function (TANH/RELU/LEAKYRELU/SWISH, default TANH):");
        String activationStr = scanner.nextLine().trim().toUpperCase();
        try {
            activation = Activation.valueOf(activationStr);
        } catch (IllegalArgumentException e) {
            System.out.println("Invalid activation function entered. Defaulting to TANH.");
            activation = Activation.TANH;
        }
        
        System.out.println("Enter weight initialization (XAVIER/HE/NORMAL, default XAVIER):");
        String weightInitStr = scanner.nextLine().trim().toUpperCase();
        try {
            weightInit = WeightInit.valueOf(weightInitStr);
        } catch (IllegalArgumentException e) {
            weightInit = WeightInit.XAVIER;
        }
        
        return new UserInput(dataPath, seed, batchSize, learningRate, epochs, 
                        neuronsPerLayer, activation, weightInit);
    }
}
