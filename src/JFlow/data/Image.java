package JFlow.data;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Function;

import javax.imageio.ImageIO;

public class Image {
    private double[][][] raw;
    private int yData;
    private double[][][] xData;

    
    protected Image(String path, int label, boolean grayscale) {
        try {
            BufferedImage img = ImageIO.read(new File(path));

            // 1 channel for grayscale, 3 for RGB
            if (grayscale) {
                raw = loadGrayscaleImage(img);
            } else {
                raw = loadRGBImage(img);
            }
            xData = raw;

        } catch (IOException e) {
            System.err.println("Error loading image: " + e.getMessage());
        }
        yData = label;
    }
    // Flattened image from csv
    protected Image(double[] image, int label) {
        int size = (int)Math.pow(image.length, 0.5);
        raw = new double[1][size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                raw[0][i][j] = image[i * size + j];
            }
        }
        yData = label;
        xData = raw;
    }

    protected Image(double[][][] image, int label) {
        raw = image;

        yData = label;
        xData = raw;
    }


    // Cumulative application of transforms
    protected void applyTransform(Function<double[][][], double[][][]> transform) {
        xData = transform.apply(xData);
    }

    public double[][][] getData() {
        return xData;
    }

    public int getLabel() {
        return yData;
    }


    // Flatten the image to 1D, keeping channels separate
    public double[] getFlat() {
        int channels = xData.length;
        int height = xData[0].length;
        int width = xData[0][0].length;
        double[] flat = new double[channels * 
            height * width];
        int index = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    flat[index++] = xData[c][h][w]; 
                }
            }
        }
        return flat;
    }

    public double[][][] getPixels() {
        return raw;
    }

    public int getWidth() {
        return xData[0].length;
    }

    public int numChannels() {
        return xData.length;
    }

    public int getHeight() {
        return xData[0][0].length;
    }


    // Return true if an image is grayscale
    // private boolean grayscaleCheck(BufferedImage img) {
    //     img = ensureRGB(img);
    //     int width = img.getWidth();
    //     int height = img.getHeight();
    //     double totalDifference = 0;
    //     int numPixels = width * height;
    
    //     for (int y = 0; y < height; y++) { 
    //         for (int x = 0; x < width; x++) { 
    //             int argb = img.getRGB(x, y);
    //             int red   = (argb >> 16) & 0xFF;
    //             int green = (argb >> 8)  & 0xFF;
    //             int blue  = (argb)       & 0xFF;
    
    //             // Compute difference from average grayscale value
    //             int avg = (red + green + blue) / 3;
    //             totalDifference += Math.abs(red - avg) + Math.abs(green - avg) + Math.abs(blue - avg);
    //         }
    //     }
    
    //     // If the average deviation is very small, consider it grayscale
    //     return (totalDifference / numPixels) < 5; // Tweak threshold if needed
    // }
    // private boolean grayscaleCheck(BufferedImage img) {
    //     img = ensureRGB(img);
    //     int width = img.getWidth();
    //     int height = img.getHeight();
    
    //     for (int y = 0; y < height; y++) { 
    //         for (int x = 0; x < width; x++) { 
    //             int argb = img.getRGB(x, y);
    //             int red   = (argb >> 16) & 0xFF;
    //             int green = (argb >> 8)  & 0xFF;
    //             int blue  = (argb)       & 0xFF;
    
    //             if (red != green || green != blue) {
    //                 return false;
    //             }
    //         }
    //     }
    //     return true;
    // }
    private boolean grayscaleCheck(BufferedImage img) {
    int width = img.getWidth();
    int height = img.getHeight();
    Set<Integer> uniqueColors = new HashSet<>();

    for (int y = 0; y < height; y++) { 
        for (int x = 0; x < width; x++) { 
            int argb = img.getRGB(x, y);
            int red   = (argb >> 16) & 0xFF;
            int green = (argb >> 8)  & 0xFF;
            int blue  = (argb)       & 0xFF;

            // Store the unique grayscale intensity
            uniqueColors.add(red);

            // If we find non-gray pixels, exit early
            if (red != green || green != blue) {
                return false;
            }
        }
    }
    return true;
}


    private BufferedImage ensureRGB(BufferedImage img) {
        if (img.getType() != BufferedImage.TYPE_INT_RGB) {
            BufferedImage converted = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
            Graphics2D g2d = converted.createGraphics();
            g2d.drawImage(img, 0, 0, null);
            g2d.dispose();
            return converted;
        }
        return img;
    }
    

    private double[][][] loadGrayscaleImage(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        double[][][] grayscaleArray = new double[1][height][width]; 

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int argb = img.getRGB(x, y);
                int gray = argb & 0xFF; 
                grayscaleArray[0][y][x] = gray; // Any
            }
        }
        return grayscaleArray;
    }

    private double[][][] loadRGBImage(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        double[][][] rgbArray = new double[3][height][width]; 

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int argb = img.getRGB(x, y);
                rgbArray[0][y][x] = (argb >> 16) & 0xFF; // Red
                rgbArray[1][y][x] = (argb >> 8)  & 0xFF; // Green
                rgbArray[2][y][x] = (argb)       & 0xFF; // Blue
            }
        }
        return rgbArray;
    }
}
    

