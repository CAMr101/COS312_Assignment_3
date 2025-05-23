package com.assignment.mlp;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class GridSearchExecutor {
    // Reduce THREAD_POOL_SIZE or keep it, but use Semaphore to control active training tasks
    private static final int MAX_CONCURRENT_TRAINING_TASKS = Math.max(1, Runtime.getRuntime().availableProcessors());
    private static final int BATCH_WRITE_SIZE = 200; // Define the number of completed trials after which results are written

    private final String dataPath;
    private final int numInputs;
    private final long baseSeed;
    private final Semaphore semaphore = new Semaphore(MAX_CONCURRENT_TRAINING_TASKS); // Initialize Semaphore

    public GridSearchExecutor(String dataPath, int numInputs, long baseSeed) {
        this.dataPath = dataPath;
        this.numInputs = numInputs;
        this.baseSeed = baseSeed;
    }

    public void executeFullGridSearch() throws IOException, InterruptedException {
        List<HyperparameterConfig> allCombinations = generateAllCombinations();
        int totalTrials = allCombinations.size(); // Get the total number of trials

        if (totalTrials == 0) {
            System.out.println("No hyperparameter combinations generated. Grid search aborted.");
            return;
        }

        File resultsFile = new File("grid_search_results.csv");
        boolean fileExists = resultsFile.exists(); // Check if file already exists to decide on header writing

        try (FileWriter writer = new FileWriter(resultsFile, true)) {
            if (!fileExists) {
                writer.write("Trial,LearningRate,BatchSize,Epochs,L1Neurons,L2Neurons,L3Neurons,Activation,WeightInit,Accuracy,F1Score,TrainingTime\n");
                writer.flush();
            }

            ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
            List<Future<GridSearchResult>> submittedFutures = new ArrayList<>();
            List<GridSearchResult> resultsBuffer = new ArrayList<>();

            System.out.println("Available processors: " + Runtime.getRuntime().availableProcessors());
            System.out.println("Max concurrent training tasks: " + MAX_CONCURRENT_TRAINING_TASKS);
            System.out.println("Starting grid search with " + totalTrials + " total trials...\n"); // Added newline

            int submittedCount = 0;
            long submissionStartTime = System.currentTimeMillis();

            // --- Phase 1: Task Submission Progress ---
            System.out.println("--- Submitting Tasks ---");
            for (HyperparameterConfig config : allCombinations) {
                semaphore.acquire();
                submittedFutures.add(executor.submit(new TrainingTask(config, dataPath, numInputs, baseSeed, submittedCount + 1, semaphore)));
                submittedCount++;

                // Update submission progress
                updateSubmissionProgress(submittedCount, totalTrials, submissionStartTime);
            }
            // Ensure the final submission progress line is printed
            updateSubmissionProgress(submittedCount, totalTrials, submissionStartTime);
            System.out.println("\nAll tasks submitted. Waiting for completion...\n"); // Added newline

            // --- Phase 2: Task Processing/Completion Progress ---
            int completedCount = 0;
            long processingStartTime = System.currentTimeMillis(); // Start time for processing phase

            while (completedCount < totalTrials) {
                Iterator<Future<GridSearchResult>> iterator = submittedFutures.iterator();
                boolean taskCompletedInThisIteration = false;

                while (iterator.hasNext()) {
                    Future<GridSearchResult> future = iterator.next();
                    if (future.isDone()) {
                        try {
                            GridSearchResult result = future.get();
                            resultsBuffer.add(result);
                            completedCount++;
                            taskCompletedInThisIteration = true;

                            // Update processing progress
                            updateProcessingProgress(completedCount, totalTrials, processingStartTime);

                            if (resultsBuffer.size() >= BATCH_WRITE_SIZE) {
                                // Clear current progress line, write info, then re-print progress
                                System.out.print("\r" + " ".repeat(120) + "\r"); // Clear the line
                                writeBufferedResults(resultsBuffer, writer);
                                updateProcessingProgress(completedCount, totalTrials, processingStartTime); // Re-print progress
                            }
                        } catch (ExecutionException e) {
                            System.err.print("\r" + " ".repeat(120) + "\r"); // Clear line before error
                            System.err.println("Error in trial " + (completedCount + 1) + ": " + e.getCause().getMessage());
                            completedCount++; // Still count as processed to avoid infinite loop
                            taskCompletedInThisIteration = true;
                            updateProcessingProgress(completedCount, totalTrials, processingStartTime); // Update progress after error
                        } finally {
                            iterator.remove();
                        }
                    }
                }

                if (!taskCompletedInThisIteration && completedCount < totalTrials) {
                    Thread.sleep(50); // Small delay to prevent busy-waiting
                }
            }

            // Write any remaining buffered results
            if (!resultsBuffer.isEmpty()) {
                System.out.print("\r" + " ".repeat(120) + "\r"); // Clear line before final write message
                writeBufferedResults(resultsBuffer, writer);
            }

            System.out.println("\nGrid search completed! Results saved to grid_search_results.csv"); // Final success message

            executor.shutdown();
            try {
                if (!executor.awaitTermination(30, TimeUnit.MINUTES)) {
                    System.err.println("Executor did not terminate in the specified time.");
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                System.err.println("Executor termination interrupted: " + e.getMessage());
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }

        } catch (Exception e) {
            System.err.println("An error occurred during grid search execution: " + e.getMessage());
        }
    }

    /**
     * Updates and prints the current progress of task submission.
     */
    private void updateSubmissionProgress(int submitted, int total, long startTime) {
        if (total == 0) return;

        double percentSubmitted = (double) submitted / total * 100;

        long elapsedMillis = System.currentTimeMillis() - startTime;
        double elapsedSeconds = elapsedMillis / 1000.0;
        double elapsedMinutes = elapsedSeconds / 60.0;

        double submissionSpeed = (submitted > 0 && elapsedMinutes > 0) ? submitted / elapsedMinutes : 0.0;

        // Estimate remaining time for submission
        double estimatedRemainingMinutes;
        double estimatedRemainingHours;
        if (submitted > 0 && submissionSpeed > 0) { // Only estimate if some tasks are submitted and speed is positive
            estimatedRemainingMinutes = (total - submitted) / submissionSpeed;
            estimatedRemainingHours = estimatedRemainingMinutes / 60.0;
        } else {
            estimatedRemainingMinutes = Double.NaN;
            estimatedRemainingHours = Double.NaN;
        }

        String remainingMinutesStr = Double.isNaN(estimatedRemainingMinutes) ? "N/A" : String.format(Locale.US, "%.1f", estimatedRemainingMinutes);
        String remainingHoursStr = Double.isNaN(estimatedRemainingHours) ? "N/A" : String.format(Locale.US, "%.1f", estimatedRemainingHours);

        // Pad with spaces to clear previous line content if it was longer
        String output = String.format(Locale.US, "\rSubmission Progress: %d/%d (%.2f%%) | Speed: %.2f tasks/min | Remaining: %s min (%s hrs)",
                submitted, total, percentSubmitted,
                submissionSpeed,
                remainingMinutesStr, remainingHoursStr);

        System.out.print(output + " ".repeat(Math.max(0, 120 - output.length()))); // Ensures line is fully cleared
    }

    /**
     * Updates and prints the current progress of task processing/completion.
     */
    private void updateProcessingProgress(int completed, int total, long startTime) {
        if (total == 0) return;

        double percentCompleted = (double) completed / total * 100;

        long elapsedMillis = System.currentTimeMillis() - startTime;
        double elapsedSeconds = elapsedMillis / 1000.0;
        double elapsedMinutes = elapsedSeconds / 60.0;

        double processingSpeed = (completed > 0 && elapsedMinutes > 0) ? completed / elapsedMinutes : 0.0;

        // Estimate remaining time for processing
        double estimatedRemainingMinutes;
        double estimatedRemainingHours;
        if (completed > 0 && processingSpeed > 0) { // Only estimate if some tasks are completed and speed is positive
            estimatedRemainingMinutes = (total - completed) / processingSpeed;
            estimatedRemainingHours = estimatedRemainingMinutes / 60.0;
        } else {
            estimatedRemainingMinutes = Double.NaN;
            estimatedRemainingHours = Double.NaN;
        }

        String remainingMinutesStr = Double.isNaN(estimatedRemainingMinutes) ? "N/A" : String.format(Locale.US, "%.1f", estimatedRemainingMinutes);
        String remainingHoursStr = Double.isNaN(estimatedRemainingHours) ? "N/A" : String.format(Locale.US, "%.1f", estimatedRemainingHours);

        // Pad with spaces to clear previous line content if it was longer
        String output = String.format(Locale.US, "\rProcessing Progress: %d/%d (%.2f%%) | Speed: %.2f trials/min | Remaining: %s min (%s hrs)",
                completed, total, percentCompleted,
                processingSpeed,
                remainingMinutesStr, remainingHoursStr);

        System.out.print(output + " ".repeat(Math.max(0, 120 - output.length()))); // Ensures line is fully cleared

        // No need to add a newline here. The last call to this method (when completed == total)
        // will implicitly be followed by the "Grid search completed!" message which already has a newline.
    }

    /**
     * Writes the collected results from the buffer to the CSV file and clears the buffer.
     *
     * @param buffer The list of GridSearchResult objects to write.
     * @param writer The FileWriter instance to write to the CSV.
     * @throws IOException If an I/O error occurs during writing.
     */
    private void writeBufferedResults(List<GridSearchResult> buffer, FileWriter writer) throws IOException {
        for (GridSearchResult result : buffer) {
            writer.write(result.toCSVString() + "\n");
        }
        writer.flush();
        System.out.println("[INFO] Wrote " + buffer.size() + " results to CSV.");
        buffer.clear();
    }

    /**
     * Generates all possible combinations of hyperparameters for the grid search.
     * This method defines the search space for the MLP.
     *
     * @return A list of HyperparameterConfig objects, each representing a unique combination.
     */
    private List<HyperparameterConfig> generateAllCombinations() {
        List<HyperparameterConfig> combinations = new ArrayList<>();

        // During Testing
        // double[] learningRates = {0.001, 0.0005, 0.0001};
        // int[] batchSizes = {32, 64, 128};
        // int[] layer1NeuronsList = {32, 64, 128, 256};
        // int[] layer2NeuronsList = {16, 32, 64, 128};
        // int[] layer3NeuronsList = {8, 16, 32, 64};
        // int[] epochsList = {50, 100, 150};
        // Activation[] activations = {Activation.RELU, Activation.SWISH, Activation.LEAKYRELU, Activation.TANH};
        // WeightInit[] weightInits = {WeightInit.XAVIER};

        // Production
        double[] learningRates = {0.0005};
        int[] batchSizes = {32, 64, 128};
        int[] layer1NeuronsList = {256};
        int[] layer2NeuronsList = {128};
        int[] layer3NeuronsList = {64};
        int[] epochsList = {150};
        Activation[] activations = {Activation.RELU, Activation.TANH};
        WeightInit[] weightInits = {WeightInit.XAVIER};

        for (double lr : learningRates) {
            for (int bs : batchSizes) {
                for (int e : epochsList) {
                    for (int l1 : layer1NeuronsList) {
                        for (int l2 : layer2NeuronsList) {
                            for (int l3 : layer3NeuronsList) {
                                // Enforce decreasing neuron count in hidden layers
                                if (l1 >= l2 && l2 >= l3) {
                                    for (Activation act : activations) {
                                        for (WeightInit wi : weightInits) {
                                            combinations.add(new HyperparameterConfig(
                                                    lr, bs, e, l1, l2, l3, act, wi
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return combinations;
    }

    /**
     * Represents a single training task for a given set of hyperparameters.
     */
    private static class TrainingTask implements Callable<GridSearchResult> {
        private final HyperparameterConfig config;
        private final String dataPath;
        private final int numInputs;
        private final long baseSeed;
        private final int trial;
        private final Semaphore semaphore;

        public TrainingTask(HyperparameterConfig config, String dataPath, int numInputs, long baseSeed, int trial, Semaphore semaphore) {
            this.config = config;
            this.dataPath = dataPath;
            this.numInputs = numInputs;
            this.baseSeed = baseSeed;
            this.trial = trial;
            this.semaphore = semaphore;
        }

        @Override
        public GridSearchResult call() throws Exception {
            try {
                DataSetIterator trainData = DataLoader.loadData(dataPath, config.getBatchSize());
                MultiLayerNetwork model = buildNetworkWithConfig(config, baseSeed + trial);

                long startTime = System.currentTimeMillis();
                for (int i = 0; i < config.getEpochs(); i++) {
                    model.fit(trainData);
                    trainData.reset();
                }
                long trainingTime = (System.currentTimeMillis() - startTime) / 1000;

                // Ensure iterator is reset before evaluation
                trainData.reset();
                Evaluation eval = model.evaluate(trainData);

                return new GridSearchResult(
                        trial, config, eval.accuracy(), eval.f1(), trainingTime
                );
            } catch (Exception e) {
                throw e; // Re-throw to be caught by the outer ExecutionException
            } finally {
                semaphore.release();
            }
        }

        private MultiLayerNetwork buildNetworkWithConfig(HyperparameterConfig config, long seed) {
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .weightInit(config.getWeightInit())
                    .updater(new Adam(config.getLearningRate()))
                    .list()
                    .layer(new DenseLayer.Builder()
                            .nIn(numInputs)
                            .nOut(config.getLayer1Neurons())
                            .activation(config.getActivation())
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nIn(config.getLayer1Neurons())
                            .nOut(config.getLayer2Neurons())
                            .activation(config.getActivation())
                            .build())
                    .layer(new DenseLayer.Builder()
                            .nIn(config.getLayer2Neurons())
                            .nOut(config.getLayer3Neurons())
                            .activation(config.getActivation())
                            .build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                            .nIn(config.getLayer3Neurons())
                            .nOut(1) // Output layer for binary classification
                            .activation(Activation.SIGMOID) // SIGMOID for binary classification
                            .build());

            MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
            model.init();
            return model;
        }
    }

    /**
     * A simple data class to hold the results of a single grid search trial.
     */
    private static class GridSearchResult {
        private final int trial;
        private final HyperparameterConfig config;
        private final double accuracy;
        private final double f1Score;
        private final long trainingTime;

        public GridSearchResult(int trial, HyperparameterConfig config,
                                double accuracy, double f1Score, long trainingTime) {
            this.trial = trial;
            this.config = config;
            this.accuracy = accuracy;
            this.f1Score = f1Score;
            this.trainingTime = trainingTime;
        }

        public String toCSVString() {
            return String.format(Locale.US,
                    "%d,%.4f,%d,%d,%d,%d,%d,%s,%s,%.4f,%.4f,%d",
                    trial,
                    config.getLearningRate(),
                    config.getBatchSize(),
                    config.getEpochs(),
                    config.getLayer1Neurons(),
                    config.getLayer2Neurons(),
                    config.getLayer3Neurons(),
                    config.getActivation(),
                    config.getWeightInit(),
                    accuracy,
                    f1Score,
                    trainingTime
            );
        }
    }
}
