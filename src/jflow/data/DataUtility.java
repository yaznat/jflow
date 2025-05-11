package jflow.data;

import java.util.stream.IntStream;

class DataUtility {
    

    protected static float max(float[][][] array) {
        float max = Float.NEGATIVE_INFINITY;

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

    protected static float min(float[][][] array) {
        float min = Float.POSITIVE_INFINITY;

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

    protected static float[][][] clip(float[][][] array, float min, float max) {
        int channels = array.length;
        int height = array[0].length;
        int width = array[0][0].length;

        float[][][] result = new float[channels][height][width];

        IntStream.range(0, channels).parallel().forEach(i -> {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    
                    result[i][j][k] = Math.max(min, Math.min(max, array[i][j][k]));
                }
            }
        });
        return result;
    }

    protected static float[][][] multiply(float[][][] array, double scalar) {
        int channels = array.length;
        int height = array[0].length;
        int width = array[0][0].length;

        float fScalar = (float)scalar;

        float[][][] result = new float[channels][height][width];

        IntStream.range(0, channels).parallel().forEach(i -> {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    result[i][j][k] =  array[i][j][k] * fScalar;
                }
            }
        });
        return result;
    }

    protected static float[][][] add(float[][][] array, double scalar) {
        int channels = array.length;
        int height = array[0].length;
        int width = array[0][0].length;

        float[][][] result = new float[channels][height][width];

        float fScalar = (float)scalar;

        IntStream.range(0, channels).parallel().forEach(i -> {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    result[i][j][k] =  array[i][j][k] + fScalar;
                }
            }
        });
        return result;
    }

    // Rotate an array clockwise by 90 degrees
    public static float[][] transpose(float[][] arr) {
        int numRows = arr.length;
        int numCols = arr[0].length;
        float[][] result = new float[numCols][numRows];

        IntStream.range(0, numRows).parallel().forEach(row -> {
            for (int col = 0; col < numCols; col++) {
                result[col][row] = arr[row][col];
            }
        });

        return result;
    }

}
