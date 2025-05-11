package jflow.utils;

import jflow.data.Image;
import jflow.data.JMatrix;

public class JPlot {

    /**
     * Displays an image in a new JFrame.
     * @param image image data wrapped in an Image.
     */
    public static void displayImage(Image image) {
        displayImage(image, 1);
    }
    /**
     * Displays an image in a new JFrame with a scale factor.
     * @param image             image data wrapped in an Image.
     * @param scaleFactor       The scale factor of the display.
     */
    public static void displayImage(Image image, int scaleFactor) {
        float[][][] pixels = image.getPixels();
        
        // Convert to [size, size, channels]
        float[][][] displayImage = new float[pixels[0].length]
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

    /**
     * Displays an image in a new JFrame with a scale factor.
     * @param image             image data wrapped in a JMatrix with shape (1, channels, height, width).
     * @param scaleFactor       The scale factor of the display.
     */
    public static void displayImage(JMatrix image, int scaleFactor, String title) {
        int channels = image.channels();
        int height = image.height();
        int width = image.width();

        // Convert to [size, size, channels]
        float[][][] displayImage = new float[height][width][channels];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                // grayscale
                displayImage[j][i][0] = image.get(0, 0, i, j);
                // RGB
                if (channels == 3) {
                    displayImage[j][i][1] = image.get(0, 1, i, j);
                    displayImage[j][i][2] = image.get(0, 2, i, j);
                }
            }
        }
        // Display the image
        new ImageDisplay(displayImage, scaleFactor, title);
    }
}
