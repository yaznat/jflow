package jflow.utils;

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
}
