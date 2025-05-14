package jflow.utils;

import java.util.HashMap;
import java.util.Map;

/**
 * A compilation of evaluation functions and more.
 */
public class Metrics {
    /**
     * Get the accuracy of model predictions.
     * @param predictions               The predicted class labels
     * @param labels                    The true class labels
     * @return                          Accuracy percentage in the range (0.0 to 1.0).
     */
    public static double getAccuracy(int[] predictions, int[] labels) {
        double sum = 0;
        for (int x = 0; x < predictions.length; x++) {
            if (predictions[x] == labels[x]) {
                sum++;
            }
        }
        return sum / predictions.length;
    }

    /**
     * Display a confusion matrix in a new JFrame
     * @param predictions               The predicted class labels
     * @param labels                    The true class labels
     */
    public static void displayConfusionMatrix(int[] predictions, int[] labels) {
        new ConfusionMatrix(predictions, labels);
    }


    /**
     * Prints out stylish training status with \r.
     * 
     * @param currentEpoch              The current epoch in training.
     * @param totalEpochs               The total number of epochs in training.
     * @param currentBatch              The current batch in training.
     * @param totalBatches              The total number of batches in training.
     * @param elapsedTime               Time elapsed since the start of training.
     * @param losses                    A hashmap with keys representing the loss names and values the loss values.
     */
    public static void printTrainingCallback(int currentEpoch, int totalEpochs, 
        int currentBatch, int totalBatches, long elapsedTime, 
        HashMap<String, Double> losses) {

        doTrainingCallback(currentEpoch, totalEpochs,
            currentBatch, totalBatches, elapsedTime, losses, null);
    }

        /**
     * Prints out stylish training status with \r.
     * 
     * @param currentEpoch              The current epoch in training.
     * @param totalEpochs               The total number of epochs in training.
     * @param currentBatch              The current batch in training.
     * @param totalBatches              The total number of batches in training.
     * @param elapsedTime               Time elapsed since the start of training.
     * @param losses                    A hashmap with keys representing the loss names and values the loss values.
     * @param learningRate              The current learning rate.
     */
    public static void printTrainingCallback(int currentEpoch, int totalEpochs, 
        int currentBatch, int totalBatches, long elapsedTime, 
        HashMap<String, Double> losses, Double learningRate) {

        doTrainingCallback(currentEpoch, totalEpochs,
            currentBatch, totalBatches, elapsedTime, losses, learningRate);
    }

    private static void doTrainingCallback(int currentEpoch, int totalEpochs, 
        int currentBatch, int totalBatches, long elapsedTime, 
        HashMap<String, Double> losses, Double learningRate) {
        
        long timePerBatch = elapsedTime / (currentBatch + 1);
        long timeRemaining = timePerBatch * (totalBatches - currentBatch);

        final String TEAL = "\033[38;2;0;153;153;1m";
        final String ORANGE = "\033[38;2;255;165;1m";
        final String WHITE = "\033[37m";
        final String RESET = "\033[0m";
        final String BOLD = "\033[1m";
        final String SEPARATOR = TEAL + " | " + RESET;

        // Replace the last line in the terminal
        String report = "\r";

        // Add epochs and batches
        report += BOLD + ORANGE + "Epoch: " + RESET + WHITE + currentEpoch + "/" + totalEpochs + 
            SEPARATOR + BOLD + ORANGE + "Batch: " + RESET + WHITE + currentBatch + "/" + totalBatches;

        // Report learning rate if applicable
        if (learningRate != null) {
            report += SEPARATOR + "lr: " + learningRate;
        }

        // Report losses if applicable
        if (losses != null) {
            for (Map.Entry<String, Double> entry : losses.entrySet()) {
                String lossName = entry.getKey();       
                float value = (float)(double)entry.getValue(); 
    
                String loss = String.valueOf(value);

                // Keep length consistent so reports aren't jumpy
                while (loss.length() < 6) {
                    loss += "0";
                }
                loss = loss.substring(0, 6);

                report += SEPARATOR + BOLD + ORANGE + lossName + ": " + RESET + WHITE + loss + RESET;
            }
        }
        // Report ETA
        report += SEPARATOR + BOLD + ORANGE + "ETA: " + RESET + WHITE + secondsToClock(
            (int)(timeRemaining * 0.000000001)) + RESET;
        
        System.out.print(report);
    }

        

    private static String secondsToClock(int totalSeconds) {
        int hours = 0; int minutes = 0;
        // hours
        if (totalSeconds > 3600) {
            int hoursDiv = totalSeconds / 3600;
            totalSeconds -= 3600 * hoursDiv;
            hours += hoursDiv;
        } else if (totalSeconds == 3600) {
            hours++;
            totalSeconds = 0;
        }
        // minutes
        if (totalSeconds > 60) {
            int minutesDiv = totalSeconds / 60;
            totalSeconds -= 60 * minutesDiv;
            minutes += minutesDiv;
        } else if (totalSeconds == 60) {
            minutes++;
            totalSeconds = 0;
        }
        if (hours != 0) {
            return hours + ":" + ((minutes < 10) ? "0" + minutes : "" + minutes);
        }
        return ((minutes < 10) ? "0" + minutes : "" + minutes) + ":" + 
            ((totalSeconds < 10) ? "0" + totalSeconds : "" + totalSeconds);
    }

}
