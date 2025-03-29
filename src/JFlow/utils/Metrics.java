package JFlow.utils;

public class Metrics {
    // Return model accuracy in range [0.0, 1.0]
    public static double getAccuracy(int[] predictions, int[] labels) {
        double sum = 0;
        for (int x = 0; x < predictions.length; x++) {
            if (predictions[x] == labels[x]) {
                sum++;
            }
        }
        return sum / predictions.length;
    }

     // Opens a JFrame with a confusion matrix
    public static void displayConfusionMatrix(int[] predictions, int[] labels) {
        new ConfusionMatrix(predictions, labels);
    }
}
