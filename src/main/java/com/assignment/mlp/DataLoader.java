package com.assignment.mlp;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
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
        System.out.println("Enter the path to the CSV file (relative or absolute):");
        String dataPath = scanner.nextLine().trim();
        
        System.out.println("Enter a random seed (default 527):");
        long seed = scanner.hasNextLong() ? scanner.nextLong() : 527;
        scanner.nextLine(); // consume newline

        System.out.println("Enter batch size (default 32):");
        int batchSize = scanner.hasNextInt() ? scanner.nextInt() : 32;
        scanner.nextLine(); // consume newline

        System.out.println("Enter learning rate (default 0.001):");
        double learningRate = scanner.hasNextDouble() ? scanner.nextDouble() : 0.001;
        scanner.nextLine(); // consume newline
        
        return new UserInput(dataPath, seed, batchSize, learningRate);
    }
}