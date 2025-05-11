package jflow.data;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.function.Function;

import javax.imageio.ImageIO;

public class Image {
    private float[][][] originalImage;
    private int yData, channels;
    private float[][][] xData;
    private boolean grayscale, lowMemoryMode, loadedFromCSV = false;
    private String path;
    private ArrayList<Function<float[][][], float[][][]>> transforms = 
    new ArrayList<Function<float[][][], float[][][]>>();


    /*
     * grayscaleCheck is currently buggy, so the user
     * must denote whether a directory is grayscale
     * or RGB.
     */
    protected Image(String path, int label, boolean grayscale, boolean lowMemoryMode) {
        this.path = path;
        this.grayscale = grayscale;
        this.channels = (grayscale) ? 1 : 3;
        this.lowMemoryMode = lowMemoryMode;
        yData = label;
    }
    // Flattened image from csv
    protected Image(float[] image, int label) {
        this.channels = 1;
        this.loadedFromCSV = true;
        int size = (int)Math.pow(image.length, 0.5);
        originalImage = new float[1][size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                originalImage[0][i][j] = image[i * size + j];
            }
        }
        yData = label;
    }

    protected Image(float[][][] image, int label) {
        this.channels = image.length;
        originalImage = image;
        yData = label;
    }

    private void load() {
        try {
            BufferedImage img = ImageIO.read(new File(path));
            // 1 channel for grayscale, 3 for RGB
            if (grayscale) {
                originalImage = loadGrayscaleImage(img);
            } else {
                originalImage = loadRGBImage(img);
            }
            xData = originalImage;

        } catch (IOException e) {
            System.err.println("Error loading image: " + e.getMessage());
        }
    }

    private void applyTransforms() {
        for (Function<float[][][], float[][][]> transform : transforms) {
            xData = transform.apply(xData);
        }
    }

    protected void addTransform(Function<float[][][], float[][][]> transform) {
        transforms.add(transform);
    }

    public float[][][] getData() {
        if (xData == null) {
            // CSV images can't be reloaded
            if (loadedFromCSV) {
                xData = originalImage;
            } else {
                load();
            }
            applyTransforms();
        }
        return xData;
    }

    public float getPixel(int flatIndex) {
        int height = xData[0].length;
        int width = xData[0][0].length;
        int channelSize = height * width;

        int channelIndex = flatIndex / channelSize;
        int reuse = flatIndex % channelSize;
        int heightIndex = reuse / width;
        int widthIndex = reuse % width;

        return xData[channelIndex][heightIndex][widthIndex];
    }

    public int getLabel() {
        return yData;
    }



    // Flatten the image to 1D, keeping channels separate
    public float[] getFlat() {
        if (xData == null) {
            // CSV images can't be reloaded
            if (loadedFromCSV) {
                xData = originalImage;
            } else {
                load();
            }
            applyTransforms();
        }
        int channels = xData.length;
        int height = xData[0].length;
        int width = xData[0][0].length;
        float[] flat = new float[channels * 
            height * width];
        int index = 0;
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    flat[index++] = xData[c][h][w]; 
                }
            }
        }
        if (lowMemoryMode) {
            unload();
        }
        return flat;
    }

    public float[][][] getPixels() {
        if (originalImage == null) {
            load();
        }
        return originalImage;
    }

    public int getWidth() {
        if (xData == null) {
            // CSV images can't be reloaded
            if (loadedFromCSV) {
                xData = originalImage;
            } else {
                load();
            }
            applyTransforms();
        }
        return xData[0][0].length;
    }

    public int numChannels() {
        return channels;
    }

    public int getHeight() {
        if (xData == null) {
            // CSV images can't be reloaded
            if (loadedFromCSV) {
                xData = originalImage;
            } else {
                load();
            }
            applyTransforms();
        }
        return xData[0].length;
    }

    private void unload() {
        originalImage = xData = null;
    }

    // Return true if an image is grayscale, CURRENTLY BUGGY
    // private boolean grayscaleCheck(BufferedImage img) {
    //     int width = img.getWidth();
    //     int height = img.getHeight();
    //     Set<Integer> uniqueColors = new HashSet<>();

    //     for (int y = 0; y < height; y++) { 
    //         for (int x = 0; x < width; x++) { 
    //             int argb = img.getRGB(x, y);
    //             int red   = (argb >> 16) & 0xFF;
    //             int green = (argb >> 8)  & 0xFF;
    //             int blue  = (argb)       & 0xFF;

    //             // Store the unique grayscale intensity
    //             uniqueColors.add(red);

    //             // If we find non-gray pixels, exit early
    //             if (red != green || green != blue) {
    //                 return false;
    //             }
    //         }
    //     }
    //     return true;
    // }


    // Load an image as grayscale: (1, height, width)
    private float[][][] loadGrayscaleImage(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        float[][][] grayscaleArray = new float[1][height][width]; 

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int argb = img.getRGB(x, y);
                int gray = argb & 0xFF; 
                grayscaleArray[0][y][x] = gray; // Any
            }
        }
        return grayscaleArray;
    }

    // Load an image as RGB: (3, height, width)
    private float[][][] loadRGBImage(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        float[][][] rgbArray = new float[3][height][width]; 

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
    

