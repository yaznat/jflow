package JFlow.data;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;
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

        int numThreads = Runtime.getRuntime().availableProcessors();
        int minSizeForParallel = 10000 * numThreads;
        double[][] result = new double[numCols][numRows];


        if (numRows * numCols >= minSizeForParallel) {
            ForkJoinPool pool = new ForkJoinPool(numThreads);
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int i = 0; i < numRows; i++) {
                final int row = i;
                tasks.add(() -> {
                    for (int col = 0; col < numCols; col++) {
                        result[col][row] = arr[row][col];
                    }
                    return null;
                });
            }
            try {
                pool.invokeAll(tasks);
            } catch (Exception e) {
                Thread.currentThread().interrupt(); 
                e.printStackTrace(); 
            }
            pool.close();
        }  else {
            for (int row = 0; row < numRows; row++) {
                for (int col = 0; col < numCols; col++) {
                    result[col][row] = arr[row][col];
                }
            }
        }
        return result;
    }

}
