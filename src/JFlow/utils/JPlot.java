package JFlow.utils;

import java.util.Arrays;

import JFlow.data.Image;

public class JPlot {

    // Opens a JFrame with an image
    public static void displayImage(Image image) {
        displayImage(image, 1);
    }
    // Optional scale factor
    public static void displayImage(Image image, int scaleFactor) {
        double[][][] pixels = image.getPixels();
        
        // Convert to [size, size, channels]
        double[][][] displayImage = new double[pixels[0].length]
            [pixels[0][0].length][pixels.length];

        for (int i = 0; i < pixels[0].length; i++) {
            for (int j = 0; j < pixels[0][0].length; j++) {
                // grayscale
                displayImage[j][i][0] = pixels[0][i][j];
                // RGB
                if (pixels.length == 3) {
                    displayImage[j][i][1] = pixels[1][i][j];
                    displayImage[j][i][2] = pixels[2][i][j];
                }
            }
        }
        // Display the image
        new ImageDisplay(displayImage, scaleFactor, String.valueOf(image.getLabel()));
    }

    // Display an image directly from an array
    public static void displayImage(double[] image, int scaleFactor, int channels, String title) {
        int size = (int)Math.sqrt(image.length / channels);
        double[][][] displayImage = new double[size][size][channels];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < channels; k++) {
                    displayImage[i][j][k] = (image[k * size * size + i * size + j] + 1) * 127.5;
                }
            }
        }
        new ImageDisplay(displayImage, scaleFactor, title);
    }
}
