package JFlow.data;

import java.util.stream.IntStream;

class DataUtility {
    

    protected static double max(double[][][] array) {
        double max = Double.NEGATIVE_INFINITY;

        int channels = array.length;
        int height = array[0].length;
        int width = array[0][0].length;

        for (int i = 0; i < channels; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    max = Math.max(max, array[i][j][k]);
                }
            }
        }

        return max;
    }

    protected static double min(double[][][] array) {
        double min = Double.POSITIVE_INFINITY;

        int channels = array.length;
        int height = array[0].length;
        int width = array[0][0].length;

        for (int i = 0; i < channels; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    min = Math.min(min, array[i][j][k]);
                }
            }
        }

        return min;
    }

    protected static double[][][] clip(double[][][] array, double min, double max) {
        int channels = array.length;
        int height = array[0].length;
        int width = array[0][0].length;

        double[][][] result = new double[channels][height][width];

        IntStream.range(0, channels).parallel().forEach(i -> {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    
                    result[i][j][k] = Math.max(min, Math.min(max, array[i][j][k]));
                }
            }
        });
        return result;
    }

    protected static double[][][] multiply(double[][][] array, double scalar) {
        int channels = array.length;
        int height = array[0].length;
        int width = array[0][0].length;

        double[][][] result = new double[channels][height][width];

        IntStream.range(0, channels).parallel().forEach(i -> {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    result[i][j][k] =  array[i][j][k] * scalar;
                }
            }
        });
        return result;
    }

    protected static double[][][] add(double[][][] array, double scalar) {
        int channels = array.length;
        int height = array[0].length;
        int width = array[0][0].length;

        double[][][] result = new double[channels][height][width];

        IntStream.range(0, channels).parallel().forEach(i -> {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    result[i][j][k] =  array[i][j][k] + scalar;
                }
            }
        });
        return result;
    }

    // Rotate an array clockwise by 90 degrees
    public static double[][] transpose(double[][] arr) {
        int numRows = arr.length;
        int numCols = arr[0].length;
        double[][] result = new double[numCols][numRows];

        IntStream.range(0, numRows).parallel().forEach(row -> {
            for (int col = 0; col < numCols; col++) {
                result[col][row] = arr[row][col];
            }
        });

        return result;
    }

}
