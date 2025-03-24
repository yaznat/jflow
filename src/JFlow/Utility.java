package JFlow;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.stream.IntStream;

public class Utility {
    
    // public static double[][] subtract(double[][] arr1, double[][] arr2) {
    //     int rows = arr1.length, cols = arr1[0].length;
    //     double[][] result = new double[rows][cols];

    //     ForkJoinPool pool = new ForkJoinPool();
    //     pool.invoke(new SubtractTask(arr1, arr2, result, 0, rows, cols));
    //     return result;
    // }
    // Concatenate arrays by subtraction
    public static double[][] subtract(double[][] arr1, double[][] arr2) throws IllegalArgumentException {
        if (arr1.length != arr2.length || arr1[0].length != arr2[0].length) {
            String errorMessage = "Arrays must have the same dimensions, not " +  "(" 
                                  + arr1.length + "," + arr1[0].length + ") and (" + arr2.length +
                                  "," + arr2[0].length + ")";
            throw new IllegalArgumentException(errorMessage);
        }

        int numRows = arr1.length;
        int numCols = arr1[0].length;
        double[][] result = new double[numRows][numCols];

        int numThreads = Runtime.getRuntime().availableProcessors();
        int minSizeForParallel = 10000 * numThreads;

        if (numRows * numCols >= minSizeForParallel) {
            ForkJoinPool pool = new ForkJoinPool();
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int i = 0; i < numRows; i++) {
                final int row = i;
                tasks.add(() -> {
                    for (int col = 0; col < numCols; col++) {
                        result[row][col] = arr1[row][col] - arr2[row][col];
                    }
                    return null;
                });
            }
            try {
                pool.invokeAll(tasks);
            } catch (Exception e) {
                Thread.currentThread().interrupt(); // Restore the interrupted status
                e.printStackTrace(); // Log the error
            }
            pool.close();
        }  else {
            for (int row = 0; row < numRows; row++) {
                for (int col = 0; col < numCols; col++) {
                    result[row][col] = arr1[row][col] - arr2[row][col];
                }
            }
        }
        return result;
    }
    // Concatenate arrays by subtraction
    public static double[][] add(double[][] arr1, double[][] arr2) throws IllegalArgumentException {
        if (arr1.length != arr2.length || arr1[0].length != arr2[0].length) {
            String errorMessage = "Arrays must have the same dimensions, not " +  "(" 
                                  + arr1.length + "," + arr1[0].length + ") and (" + arr2.length +
                                  "," + arr2[0].length + ")";
            throw new IllegalArgumentException(errorMessage);
        }

        int numRows = arr1.length;
        int numCols = arr1[0].length;
        double[][] result = new double[numRows][numCols];

        int numThreads = Runtime.getRuntime().availableProcessors();
        int minSizeForParallel = 10000 * numThreads;

        if (numRows * numCols >= minSizeForParallel) {
            ForkJoinPool pool = new ForkJoinPool();
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int i = 0; i < numRows; i++) {
                final int row = i;
                tasks.add(() -> {
                    for (int col = 0; col < numCols; col++) {
                        result[row][col] = arr1[row][col] + arr2[row][col];
                    }
                    return null;
                });
            }
            try {
                pool.invokeAll(tasks);
            } catch (Exception e) {
                Thread.currentThread().interrupt(); // Restore the interrupted status
                e.printStackTrace(); // Log the error
            }
            pool.close();
        }  else {
            for (int row = 0; row < numRows; row++) {
                for (int col = 0; col < numCols; col++) {
                    result[row][col] = arr1[row][col] + arr2[row][col];
                }
            }
        }
        return result;
    }
    // Broadcast multiply arrays
    public static double[][] multiply(double[][] arr1, double[][] arr2) throws IllegalArgumentException {
        if (arr1.length != arr2.length || arr1[0].length != arr2[0].length) {
            String errorMessage = "Arrays must have the same dimensions, not " +  "(" 
                                  + arr1.length + "," + arr1[0].length + ") and (" + arr2.length +
                                  "," + arr2[0].length + ")";
            throw new IllegalArgumentException(errorMessage);
        }

        int numRows = arr1.length;
        int numCols = arr1[0].length;
        double[][] result = new double[numRows][numCols];

        int numThreads = Runtime.getRuntime().availableProcessors();
        int minSizeForParallel = 10000 * numThreads;

        if (numRows * numCols >= minSizeForParallel) {
            ForkJoinPool pool = new ForkJoinPool();
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int i = 0; i < numRows; i++) {
                final int row = i;
                tasks.add(() -> {
                    for (int col = 0; col < numCols; col++) {
                        result[row][col] = arr1[row][col] * arr2[row][col];
                    }
                    return null;
                });
            }
            try {
                pool.invokeAll(tasks);
            } catch (Exception e) {
                Thread.currentThread().interrupt(); // Restore the interrupted status
                e.printStackTrace(); // Log the error
            }
            pool.close();
        }  else {
            for (int row = 0; row < numRows; row++) {
                for (int col = 0; col < numCols; col++) {
                    result[row][col] = arr1[row][col] * arr2[row][col];
                }
            }
        }
        return result;
    }
    // Multiply an array by a scalar
    public static double[][] multiply(double[][] arr, double scalar) throws IllegalArgumentException {
        int numRows = arr.length;
        int numCols = arr[0].length;

        int numThreads = Runtime.getRuntime().availableProcessors();
        int minSizeForParallel = 10000 * numThreads;
        double[][] result = new double[numRows][numCols];


        if (numRows * numCols >= minSizeForParallel) {
            ForkJoinPool pool = new ForkJoinPool();
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int i = 0; i < numRows; i++) {
                final int row = i;
                tasks.add(() -> {
                    for (int col = 0; col < numCols; col++) {
                        result[row][col] = arr[row][col] * scalar;
                    }
                    return null;
                });
            }
            try {
                pool.invokeAll(tasks);
            } catch (Exception e) {
                Thread.currentThread().interrupt(); // Restore the interrupted status
                e.printStackTrace(); // Log the error
            }
            pool.close();
        }  else {
            for (int row = 0; row < numRows; row++) {
                for (int col = 0; col < numCols; col++) {
                    result[row][col] = arr[row][col] * scalar;
                }
            }
        }
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
                Thread.currentThread().interrupt(); // Restore the interrupted status
                e.printStackTrace(); // Log the error
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

    public static double sum(double[][] arr) {
        double sum = 0;
        for (int x = 0; x < arr.length; x++) {
            for (int y = 0; y < arr[0].length; y++) {
                sum += arr[x][y];
            }
        }
        return sum;
    }
    public static double[] sum0(double[][] arr, boolean scale) {
        double[] sum = new double[arr.length]; 
        double scaleFactor = (scale) ? 1.0 / arr[0].length : 1;
        
        for (int y = 0; y < arr.length; y++) { 
            for (int x = 0; x < arr[0].length; x++) { 
                sum[y] += arr[y][x] * scaleFactor; 
            }
        }
    
        return sum;
    }

    // To automate the label reading process
    public static int max(int[] arr) {
        int max = 0;
        for (int i : arr) {
            max = Math.max(max, i);
        }
        return max;
    }
    public static int max(int[][] arr) {
        int max = 0;
        for (int[] i : arr) {
            for (int i2 : i) {
                max = Math.max(max, i2);
            }
        }
        return max;
    }

    public static double max(double[] arr) {
        double max = Double.NEGATIVE_INFINITY;
        for (double d : arr) {
            max = Math.max(max, Math.abs(d));
        }
        return max;
    }

    public static double max(double[][][] arr) {
        double max = Double.NEGATIVE_INFINITY;
        for (double[][] d : arr) {
            for (double[] d1: d) {
                for (double d2: d1) {
                    max = Math.max(max, d2);
                } 
            }
        }
        return max;
    }

    public static double max(double[][][][] arr) {
        double max = Double.NEGATIVE_INFINITY;
        for (double[][][] d : arr) {
            for (double[][] d1: d) {
                for (double[] d2: d1) {
                    for (double d3: d2) {
                        max = Math.max(max, d3);
                    }
                }
            }
        }
        return max;
    }

    public static double min(double[][][][] arr) {
        double min = Double.POSITIVE_INFINITY;
        for (double[][][] d : arr) {
            for (double[][] d1: d) {
                for (double[] d2: d1) {
                    for (double d3: d2) {
                        min = Math.min(min, d3);
                    }
                }
            }
        }
        return min;
    }

    public static double mean(double[][][][] arr) {
        double mean = 0;
        
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                for (int k = 0; k < arr[0][0].length; k++) {
                    for (int l = 0; l < arr[0][0][0].length; l++) {
                        mean += arr[i][j][k][l];
                    }
                }
            }
        }

        return mean / (arr.length * arr[0].length * arr[0][0].length * arr[0][0][0].length);
    }

    public static double variance(double[][][][] arr) {
        double mean = mean(arr);
        double variance = 0;
        
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                for (int k = 0; k < arr[0][0].length; k++) {
                    for (int l = 0; l < arr[0][0][0].length; l++) {
                        variance += Math.pow(arr[i][j][k][l] - mean, 2);
                    }
                }
            }
        }

        return variance / (arr.length * arr[0].length * arr[0][0].length * arr[0][0][0].length);
    }
    
    public static double[] subtract(double[] arr1, double[] arr2) {
        if (arr1.length != arr2.length) {
            throw new IllegalArgumentException("Array size must match. Got " 
                + arr1.length + " and " + arr2.length);
        }
        double[] result = new double[arr1.length];
        for (int i = 0; i < arr1.length; i++) {
            result[i] = arr1[i] - arr2[i];
        }
        return result;
    }

    public static double[] add(double[] arr1, double[] arr2) {
        if (arr1.length != arr2.length) {
            throw new IllegalArgumentException("Array size must match. Got " 
                + arr1.length + " and " + arr2.length);
        }
        double[] result = new double[arr1.length];
        for (int i = 0; i < arr1.length; i++) {
            result[i] = arr1[i] + arr2[i];
        }
        return result;
    }

    public static double[] multiply(double[] arr, double scalar) {
        ForkJoinPool pool = new ForkJoinPool();
        double[] result = new double[arr.length];
        pool.submit(() -> 
            IntStream.range(0, arr.length).parallel().forEach(i -> 
                result[i] = arr[i] * scalar
            )
        ).join();
        pool.close();
        return result;
    }

    public static double[] multiply(double[] arr1, double[] arr2) {
        ForkJoinPool pool = new ForkJoinPool();
        double[] result = new double[arr1.length];
        pool.submit(() -> 
            IntStream.range(0, arr1.length).parallel().forEach(i -> 
                result[i] = arr1[i] * arr2[i]
            )
        ).join();
        pool.close();
        return result;
    }
    


    public static double max(double[][] arr) {
        double max = Double.NEGATIVE_INFINITY;
        for (double[] d : arr) {
            for (double d1: d) {
                max = Math.max(max, Math.abs(d1));
            }
        }
        return max;
    }

    public static double[][] clip(double[][] arr, double min, double max) {
        double[][] result = new double[arr.length][arr[0].length];
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[0].length; j++) {
                result[i][j] = Math.max(min, Math.min(max, arr[i][j]));
            }
        }
        return result;
    }
    public static double[] clip(double[] arr, double min, double max) {
        double[] result = new double[arr.length];

        Arrays.parallelSetAll(result, i -> Math.max(min, Math.min(max, arr[i])));

        return result;
    }

    public static double[][] conv2D(double[][][] image, double[][][] filter, String padding) {
        int inputDepth = image.length;
        int inputHeight = image[0].length;
        int inputWidth = image[0][0].length;

        int filterDepth = filter.length;
        int filterHeight = filter[0].length;
        int filterWidth = filter[0][0].length;

        int outputHeight, outputWidth;
        double[][][] tensor;

        // Determine output size
        if (padding.equals("same_padding")) {
            int p = (int)((filterHeight - 1) / 2.0);
            outputHeight = inputHeight;
            outputWidth = inputWidth;

            // Create padded tensor
            tensor = new double[inputDepth][inputHeight + 2 * p][inputWidth + 2 * p];
            for (int d = 0; d < inputDepth; d++) {
                for (int i = 0; i < inputHeight; i++) {
                    for (int j = 0; j < inputWidth; j++) {
                        tensor[d][i + p][j + p] = image[d][i][j];
                    }
                }
            }
        } else {
            outputHeight = inputHeight - filterHeight + 1;
            outputWidth = inputWidth - filterWidth + 1;
            tensor = image;
        }
        // Initialize resulting image
        double[][] result = new double[outputHeight][outputWidth];
        // Perform convolution
        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                double sum = 0;
                for (int d = 0; d < filterDepth; d++) {
                    for (int a = 0; a < filterHeight; a++) {
                        for (int b = 0; b < filterWidth; b++) {
                            sum += tensor[d][i + a][j + b] * filter[d][a][b];
                        }
                    }
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    public static double[][] maxPool2D(double[][] image, int poolSize, int stride) {
        // Initialize resulting image
        int outputSize = (int)(image.length / (double)poolSize);
        double[][] result = new double[outputSize][outputSize];

        // Perform pooling
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                // Find max value per pool
                double maxValue = Double.NEGATIVE_INFINITY;

                for (int a = 0; a < stride; a++) {
                    for (int b = 0; b < stride; b++) {
                        double pixel = image[i * stride + a][j * stride + b];
                        if (pixel > maxValue) {
                            maxValue = pixel;
                        }
                    }
                }
                result[i][j] = maxValue;
            }
        }
        return result;
    }

    public static double[] flatten2DArray(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;
        double[] flattened = new double[rows * cols];

        ForkJoinPool pool = new ForkJoinPool(); // Use default parallelism level
        int numThreads = pool.getParallelism();
        ForkJoinTask<?>[] tasks = new ForkJoinTask<?>[numThreads];

        for (int t = 0; t < numThreads; t++) {
            final int threadId = t;
            tasks[t] = pool.submit(() -> {
                for (int i = threadId; i < rows; i += numThreads) {
                    int startIdx = i * cols;
                    System.arraycopy(array[i], 0, flattened, startIdx, cols);
                }
            });
        }

        // Wait for all tasks to complete
        for (ForkJoinTask<?> task : tasks) {
            task.join();
        }

        pool.shutdown();
        pool.close();
        return flattened;
    }

}
