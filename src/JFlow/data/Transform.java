package JFlow.data;
import java.util.ArrayList;
import java.util.function.Function;

import JFlow.Utility;

public class Transform {
    // Store transforms
    private ArrayList<Function<double[][][], double[][][]>> transforms;
    public Transform() {
        transforms = new ArrayList<Function<double[][][], double[][][]>>();
    }

    protected ArrayList<Function<double[][][], double[][][]>> getTransforms() {
        return transforms;
    }

    // Normalize data to [0, 1]
    public void normalizeSigmoid() {

        transforms.add(
            image -> {
                return DataUtility.multiply(image, 1.0 / 255);

            }
        );
    }

    // Normalize data to [-1, 1]
    public void normalizeTanH() {

        transforms.add(
            image -> {
                return DataUtility.add(DataUtility.multiply(image, 1 / 127.5), -1);
            }
        );
    }

    // Apply convolutional preprocess
    public void convolutionalPreprocess() {
        transforms.add(
            image -> {
                return convolutionalPreprocess(image);
            }
        );
    }

    // Rotate image: 90, 180, or 270 degrees
    public void randomRotation() {
        transforms.add(
            image -> {
                int channels = image.length;
                int numRotations = (int)(Math.random() * 3) + 1;
                double[][][] rotatedImage = copy(image);
                for (int c = 0; c < channels; c++) {
                    for (int r = 0; r < numRotations; r++) {
                        rotatedImage[c] = Utility.transpose(rotatedImage[c]);
                    }
                }
                return rotatedImage;
            }
        );
    }

    // Temporary simplified version for increasing size
    public void resize(int height, int width) {
        transforms.add(
            image -> {
                int oldHeight = image[0].length;
                int oldWidth = image[0][0].length;
                int channels = image.length;
                double[][][] resized = new double[channels][height][width];
                for (int c = 0; c < channels; c++) {
                    for (int h = 0; h < oldHeight; h++) {
                        for (int w = 0; w < oldWidth; w++) {
                            resized[c][h][w] = image[c][h][w];
                        }
                    }
                }
                return resized;
            }
        );
    }

    private double[][][] copy(double[][][] arr) {
        int channels = arr.length;
        int height = arr[0].length;
        int width = arr[0][0].length;
        double[][][] copy = new double[channels][height][width];
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    copy[c][h][w] = arr[c][h][w];
                }
            }
        }
        return copy;
    }
    // Convolutional and maxpool preprocess
    // Preprocess data with convolutions
    public static double[][][] convolutionalPreprocess(double[][][] input) {
        int channels = input.length;
        int imageSize = input[0].length * input[0][0].length;
        int numFilters = 10;
        // increase image dimensions
        double[][][][] images = new double[1][channels][imageSize][imageSize];
        images[0] = input;
        
        // Declare convolutional filters 
        double[][][] imageFilters = new double[][][]{

            {{-1, 0, 1},
             {-2, 0, 2},
             {-1, 0, 1}},

             {{-1, -2, -1},
             {0, 0, 0},
             {1, 2, 1}},

             {{-1, 0, 1},
             {-1, 0, 1},
             {-1, 0, 1}},

             {{-1, -1, -1},
             {0, 0, 0},
             {1, 1, 1}},

             {{-3, 10, -3},
             {0, 0, 0},
             {3, 10, 3}},

             {{-3, 0, 3},
             {-10, 0, 10},
             {-3, 0, 3}},

             {{0, -1, 0},
             {-1, 4, -1},
             {0, -1, 0}},

             {{0.0625, 0.125, 0.0625},
             {0.125, 0.25, 0.125},
             {0.0625, 0.125, 0.0625}},

             {{0, -1, 0},
             {-1, 5, -1},
             {0, -1, 0}},

             {{-1, -1, -1},
             {-1, 9, -1},
             {-1, -1, -1}},
        };
        double[][][][] filters = new double[numFilters][channels][3][3];
        for (int i = 0; i < numFilters; i++) {
            filters[i][0] = imageFilters[i];
        }

        // Apply convolutional filters
        double[][][][] conv = new double[1][numFilters][imageSize][imageSize];
        for (int k = 0; k < numFilters; k++) {
            conv[0][k] = Utility.conv2D(images[0], filters[k], "same_padding");
                    
        }

        // Apply max pooling 
        double[][][][] max = new double[1][numFilters][imageSize / 2][imageSize / 2];
        for (int k = 0; k < numFilters; k++) {
            max[0][k] = Utility.maxPool2D(conv[0][k], 2, 2);
        }

        return max[0];
    }

    // Resize images by a certain method
    public void resize(int width, int height, String method) {
        // TODO
    }

    // Flattens a 3D array to 1D
    public static double[] flatten3D(double[][][] arr) {
        int d1 = arr.length; int d2 = arr[0].length; int d3 = arr[0][0].length;
        double[] flattened = new double[d1 * d2 * d3];
        int index = 0;
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int k = 0; k < d3; k++) {
                    flattened[index] = arr[i][j][k];
                    index++;
                }
            }
        }
        return flattened;
    }

}
