// package JFlow.Layers;

// import java.util.ArrayList;
// import java.util.List;
// import java.util.Random;
// import java.util.concurrent.Callable;
// import java.util.concurrent.ExecutionException;
// import java.util.concurrent.ForkJoinPool;
// import java.util.stream.IntStream;

// import JFlow.Utility;

// public class Conv2DOld extends Layer{
//     private double[][][][] A, Z, dZ, filters, dFilters, vFilters, lastInput, dX;
//     private double[] biases, dBiases, vBiases;
//     private final double beta = 0.9; // Momentum coefficient
//     private Activation activation;
//     private int numFilters, filterSize, numChannels;
//     private String padding;
//     protected Conv2DOld(int numFilters, int inputChannels, int filterSize, String padding) {
//         super(numFilters * filterSize * filterSize, "conv_2d");
//         this.numFilters = numFilters;
//         this.filterSize = filterSize;
//         this.numChannels = inputChannels;
//         this.padding = padding;
//         // Initialize filters with He
//         Random rand = new Random();
//         filters = new double[numFilters][numChannels][filterSize][filterSize];
//         double stdDev = Math.sqrt(2.0 / (numChannels * filterSize * filterSize));

//         for (int k = 0; k < numFilters; k++) {
//             for (int c = 0; c < numChannels; c++) {
//                 for (int i = 0; i < filterSize; i++) {
//                     for (int j = 0; j < filterSize; j++) {
//                         filters[k][c][i][j] = rand.nextGaussian() * stdDev;
//                         // filters[k][c][i][j] = Math.random() - 0.5;
//                     }
//                 }
//             }
//         }
//         // Initialize biases to 0
//         biases = new double[numFilters];

//         // Initialize momentum weights and biases
//         vFilters = new double[numFilters][numChannels][filterSize][filterSize];
//         vBiases = new double[numFilters];

//     }
//     @Override
//     public void setActivation(Activation activation) {
//         this.activation = activation;
//     }

//     @Override
//     public void forward(double[][][][] input, boolean training) {
//         lastInput = input;

//         // System.out.println("Max input conv1: " + Utility.max(input)); // prints 1.0

//         // System.out.println("Max input[n] conv1: " + Utility.max(input[63])); // prints 1.0

//         int numImages = input.length;
//         int height = input[0][0].length;
//         int width = input[0][0][0].length;

//         A = new double[numImages][numFilters][height][width];
        
//         ForkJoinPool pool = ForkJoinPool.commonPool();

//         pool.submit(() -> IntStream.range(0, numImages).forEach(j -> {
                    
//                     for (int k = 0; k < numFilters; k++) {
//                         A[j][k] = conv2D(
//                             input[j],
//                             filters[k], 
//                             biases[k], padding);
//                     }
//                 })).join();

//         // System.out.println("Max A: " + Utility.max(A));

//         if (activation instanceof ReLU) {
//             Z = ReLU(A);
//         } else if (activation instanceof LeakyReLU) {
//             Z = leeakyReLU(A,((LeakyReLU)activation).getAlpha());
//         }
//         // implement other activations eventually

//         if (getNextLayer() != null) {
//             getNextLayer().forward(Z, training);
//         }
//     }

//     @Override
//     public void forward(double[][] input, boolean training) {
//         // TODO Auto-generated method stub
//         throw new UnsupportedOperationException("Unimplemented method 'forward'");
//     }

//     @Override
//     public void backward(double[][][][] input, double learningRate) {
//         int numImages = input.length;
//         int height = input[0][0].length;
//         int width = input[0][0][0].length;

//         ForkJoinPool pool = ForkJoinPool.commonPool();
//         // Calculate dActivation
//         if (activation instanceof ReLU) {
//             dZ = dReLU(input, Z);
//         } else if (activation instanceof LeakyReLU) {
//             dZ = dLeakyReLU(input, Z, ((LeakyReLU)activation).getAlpha());
//         }


//         // System.out.println("Max dZ: " + Utility.max(dZ));

//         // Calculate dFilters
//         dFilters = new double[numFilters][numChannels][filterSize][filterSize];

//         try {
//             pool.submit(() -> IntStream.range(0, numFilters).parallel().forEach(k -> {
//                 for (int c = 0; c < numChannels; c++) {
//                     double[][] accumulatedSum = new double[filterSize][filterSize];

//                     for (int i = 0; i < numImages; i++) {
//                         accumulatedSum = Utility.add(
//                             accumulatedSum, 
//                             conv2D(lastInput[i][c], dZ[i][k], 0, "full_padding")
//                         );
//                     }

//                     dFilters[k][c] = accumulatedSum;
//                 }
//             })).get();
//         } catch (InterruptedException | ExecutionException e) {
//             e.printStackTrace();
//         }
//         // System.out.println("Max dFilters: " + Utility.max(dFilters)); 

//         // Calculate dInput (dLastInput)
//         dX = new double[numImages][numChannels][height][width];

//         try {
//             pool.submit(() -> IntStream.range(0, numImages).parallel().forEach(i -> {
//                 for (int c = 0; c < numChannels; c++) {
//                     double[][] accumulatedGradients = new double[height][width];
    
//                     for (int k = 0; k < numFilters; k++) {
//                         accumulatedGradients = Utility.add(
//                             accumulatedGradients,
//                             conv2D(dZ[i][k], rotate180(filters[k][c]), 0, "same_padding")
//                         );
//                     }
    
//                     dX[i][c] = accumulatedGradients;
//                 }
//             })).get();
//         } catch (InterruptedException | ExecutionException e) {
//             e.printStackTrace();
//         }
    

//         // Apply targeted scale to combat vanishing gradients
//         // dX = normalizeAndScaleUp(dX, 10);

//         // System.out.println("Max dX: " + Utility.max(dX)); 

//         // Calculate dBiases
//         dBiases = new double[numFilters];

//         try {
//             pool.submit(() -> IntStream.range(0, numFilters).parallel().forEach(k -> {
//                 double sum = 0;
//                 for (int i = 0; i < numImages; i++) {
//                     for (int h = 0; h < height; h++) {
//                         for (int w = 0; w < width; w++) {
//                             sum += dZ[i][k][h][w];
//                         }
//                     }
//                 }
//                 dBiases[k] = sum;
//             })).get();
//         } catch (InterruptedException | ExecutionException e) {
//             e.printStackTrace();
//         }
//         // Update parameters using momentum
//         updateFiltersWithMomentum(filters, dFilters, vFilters, learningRate);
//         // biases = Utility.subtract(biases, Utility.multiply(dBiases, learningRate));
//         updateBiasesWithMomentum(biases, dBiases, vBiases, learningRate);

//         if (getPreviousLayer() != null) {
//             getPreviousLayer().backward(dX, learningRate);
//         }
//     }

//     public double[][][][] getFilters() {
//         return filters;
//     }
//     public double[] getBiases() {
//         return biases;
//     }

//     @Override
//     public void backward(double[][] input, double learningRate) {
//         // TODO Auto-generated method stub
//         throw new UnsupportedOperationException("Unimplemented method 'backward'");
//     }

//     @Override
//     public double[][] getOutput() {
//         // TODO Auto-generated method stub
//         throw new UnsupportedOperationException("Unimplemented method 'getOutput'");
//     }
    

//     // Perform convolution on a 3D image with n filters
//     private double[][] conv2D(double[][][] image, double[][][] filter, 
//                               double bias, String padding) {
//         int inputDepth = image.length;
//         int inputHeight = image[0].length;
//         int inputWidth = image[0][0].length;

//         int filterDepth = filter.length;
//         int filterHeight = filter[0].length;
//         int filterWidth = filter[0][0].length;

//         int outputHeight, outputWidth;

//         double[][][] tensor;


//         // Determine output size
//         if (padding.equals("same_padding")) {
//             int p = (filterHeight - 1) / 2;
//             outputHeight = inputHeight;
//             outputWidth = inputWidth;

//             // Create padded tensor
//             tensor = new double[inputDepth][inputHeight + 2 * p][inputWidth + 2 * p];
//             for (int d = 0; d < inputDepth; d++) {
//                 for (int i = 0; i < inputHeight; i++) {
//                     for (int j = 0; j < inputWidth; j++) {
//                         tensor[d][i + p][j + p] = image[d][i][j];
//                     }
//                 }
//             }
//         } else {
//             outputHeight = inputHeight - filterHeight + 1;
//             outputWidth = inputWidth - filterWidth + 1;
//             tensor = image;
//         }

//         // System.out.println("Max tensor value after padding: " + Utility.max(tensor));
//         // Initialize resulting image
//         double[][] result = new double[outputHeight][outputWidth];
//         // Perform convolution
//         for (int i = 0; i < outputHeight; i++) {
//             for (int j = 0; j < outputWidth; j++) {
//                 double sum = 0;
//                 for (int d = 0; d < filterDepth; d++) {
//                     for (int a = 0; a < filterHeight; a++) {
//                         for (int b = 0; b < filterWidth; b++) {
//                             sum += tensor[d][i + a][j + b] * filter[d][a][b];
//                         }
//                     }
//                 }
//                 result[i][j] = sum + bias;
//             }
//         }
//         return result;
//     }

    // // Perform convolution on a 2D image with one filter
    // private double[][] conv2D(double[][] image, double[][] filter, 
    //                           double bias, String padding) {
    //     int inputHeight = image.length;
    //     int inputWidth = image[0].length;

    //     int filterHeight = filter.length;
    //     int filterWidth = filter[0].length;

    //     int outputHeight, outputWidth = 0;

    //     double[][] tensor;

    //     // Determine output size
    //     if (padding.equals("same_padding") && inputHeight > filterHeight) {
    //         int p = (filterHeight - 1) / 2;
    //         outputHeight = inputHeight;
    //         outputWidth = inputWidth;

    //         // Create padded tensor
    //         tensor = new double[inputHeight + 2 * p][inputWidth + 2 * p];
    //         for (int i = 0; i < inputHeight; i++) {
    //             for (int j = 0; j < inputWidth; j++) {
    //                 tensor[i + p][j + p] = image[i][j];
    //             }
    //         }
    //     } else if (padding.equals("full_padding")) { 
    //         int p = filterSize - 1;
        
    //         // outputHeight = inputHeight + 2 * p - filterHeight + 1;
    //         // outputWidth = inputWidth + 2 * p - filterWidth + 1;
    //         outputHeight = filterSize; // temporary fix
    //         outputWidth = filterSize; // temporary fix
        
    //         tensor = new double[inputHeight + 2 * p][inputWidth + 2 * p];
        
    //         for (int i = 0; i < inputHeight; i++) {
    //             for (int j = 0; j < inputWidth; j++) {
    //                 tensor[i + p][j + p] = image[i][j];
    //             }
    //         }
    //     } else {
    //         outputHeight = inputHeight - filterHeight + 1;
    //         outputWidth = inputWidth - filterWidth + 1;
    //         tensor = image;
    //     }


    //     // Initialize resulting image
    //     double[][] result = new double[outputHeight][outputWidth];
    //     // Perform convolution
    //     for (int i = 0; i < outputHeight; i++) {
    //         for (int j = 0; j < outputWidth; j++) {
    //             double sum = 0;
                
    //             for (int a = 0; a < filterHeight; a++) {
    //                 for (int b = 0; b < filterWidth; b++) {
    //                     sum += tensor[i + a][j + b] * filter[a][b];
    //                 }
    //             }
    //             result[i][j] = sum + bias;
    //         }
    //     }
    //     return result;
    // }

//     private double[][][][] ReLU (double[][][][] arr) {
//         int dim1 = arr.length;
//         int dim2 = arr[0].length;
//         int dim3 = arr[0][0].length;
//         int dim4 = arr[0][0][0].length;

//         int numThreads = Runtime.getRuntime().availableProcessors();
//         int minSizeForParallel = 10000 * numThreads;
//         double[][][][] result = new double[dim1][dim2][dim3][dim4];

//         if (dim1 * dim2 * dim3 * dim4 >= minSizeForParallel) {
//             ForkJoinPool pool = ForkJoinPool.commonPool();
//             List<Callable<Void>> tasks = new ArrayList<>();
                    
//             for (int i = 0; i < dim1; i++) {
//                 final int idx1 = i;
//                 tasks.add(() -> {
//                     for (int j = 0; j < dim2; j++) {
//                         for (int k = 0; k < dim3; k++) {
//                                 for (int l = 0; l < dim4; l++) {
//                                 result[idx1][j][k][l] = Math.max(arr[idx1][j][k][l], 0);
//                             }
//                         }
//                     }
//                     return null;
//                 });
//             }
//             pool.invokeAll(tasks);
//         } else {
//             for (int i = 0; i < dim1; i++) {
//                 for (int j = 0; j < dim2; j++) {
//                     for (int k = 0; k < dim3; k++) {
//                         for (int l = 0; l < dim4; l++) {
//                             result[i][j][k][l] = Math.max(arr[i][j][k][l], 0);
//                         }
//                     }
//                 }
//             }
//         }
//         return result;
//     }

//     private double[][][][] leeakyReLU (double[][][][] arr, double alpha) {
//         int dim1 = arr.length;
//         int dim2 = arr[0].length;
//         int dim3 = arr[0][0].length;
//         int dim4 = arr[0][0][0].length;

//         int numThreads = Runtime.getRuntime().availableProcessors();
//         int minSizeForParallel = 10000 * numThreads;
//         double[][][][] result = new double[dim1][dim2][dim3][dim4];

//         if (dim1 * dim2 * dim3 * dim4 >= minSizeForParallel) {
//             ForkJoinPool pool = ForkJoinPool.commonPool();
//             List<Callable<Void>> tasks = new ArrayList<>();
                    
//             for (int i = 0; i < dim1; i++) {
//                 final int idx1 = i;
//                 tasks.add(() -> {
//                     for (int j = 0; j < dim2; j++) {
//                         for (int k = 0; k < dim3; k++) {
//                                 for (int l = 0; l < dim4; l++) {
//                                 result[idx1][j][k][l] = Math.max(arr[idx1][j][k][l], alpha);
//                             }
//                         }
//                     }
//                     return null;
//                 });
//             }
//             pool.invokeAll(tasks);
//         } else {
//             for (int i = 0; i < dim1; i++) {
//                 for (int j = 0; j < dim2; j++) {
//                     for (int k = 0; k < dim3; k++) {
//                         for (int l = 0; l < dim4; l++) {
//                             result[i][j][k][l] = Math.max(arr[i][j][k][l], alpha);
//                         }
//                     }
//                 }
//             }
//         }
//         return result;
//     }

//     public void updateFilters(double[][][][] filters, 
//         double[][][][] dFilters, double learningRate) {
//         int size = filters.length;
//         int channels = filters[0].length;
//         int height = filters[0][0].length;
//         int width = filters[0][0][0].length;
//         for (int i = 0; i < size; i++) {
//             for (int c = 0; c < channels; c++) {
//                 for (int h = 0; h < height; h++) {
//                     for (int w = 0; w < width; w++) {
//                         filters[i][c][h][w] -= dFilters[i][c][h][w] * learningRate;
//                     }
//                 }
//             }
//         }
//     }

//     // Efficient dRelu calculation avoids extra steps
//     private double[][][][] dReLU(double[][][][] dZ, double[][][][] Z) {
//         int numImages = dZ.length;
//         int channels = dZ[0].length;
//         int height = dZ[0][0].length;
//         int width = dZ[0][0][0].length;

//         ForkJoinPool pool = ForkJoinPool.commonPool();

//         double[][][][] result = new double[numImages][channels][height][width];

//         try {
//             pool.submit(() -> 
//                 IntStream.range(0, numImages).parallel().forEach(i -> {
//                     for (int j = 0; j < channels; j++) {
//                         for (int k = 0; k < height; k++) {
//                             for (int l = 0; l < width; l++) {
//                                 if (Z[i][j][k][l] > 0) {
//                                     result[i][j][k][l] = dZ[i][j][k][l];
//                                 }
//                             }
//                         }
//                     }
//                 })
//             ).get(); 
//         } catch (InterruptedException | ExecutionException e) {
//             e.printStackTrace();
//         }

//         return result;
//     }

//     // Efficient dLeakyRelu calculation avoids extra steps
//     private double[][][][] dLeakyReLU(double[][][][] dZ, double[][][][] Z, double alpha) {
//         int numImages = dZ.length;
//         int channels = dZ[0].length;
//         int height = dZ[0][0].length;
//         int width = dZ[0][0][0].length;

//         ForkJoinPool pool = ForkJoinPool.commonPool();

//         double[][][][] result = new double[numImages][channels][height][width];

//         try {
//             pool.submit(() -> 
//                 IntStream.range(0, numImages).parallel().forEach(i -> {
//                     for (int j = 0; j < channels; j++) {
//                         for (int k = 0; k < height; k++) {
//                             for (int l = 0; l < width; l++) {
//                                 if (Z[i][j][k][l] > 0) {
//                                     result[i][j][k][l] = dZ[i][j][k][l];
//                                 } else {
//                                     result[i][j][k][l] = alpha * dZ[i][j][k][l];
//                                 }
//                             }
//                         }
//                     }
//                 })
//             ).get(); 
//         } catch (InterruptedException | ExecutionException e) {
//             e.printStackTrace();
//         }

//         return result;
//     }

//     private double[][][][] normalizeAndScaleUp(double[][][][] arr, double amplification) {
//         double stdDev = Math.sqrt(Utility.variance(arr) + 1e-8);
//         double mean = Utility.mean(arr);

//         int numImages = arr.length;
//         int channels = arr[0].length;
//         int height = arr[0][0].length;
//         int width = arr[0][0][0].length;

//         double[][][][] result = new double[numImages][channels][height][width];

//         for (int i = 0; i < numImages; i++) {
//             for (int j = 0; j < channels; j++) {
//                 for (int k = 0; k < height; k++) {
//                     for (int l = 0; l < width; l++) {
//                         result[i][j][k][l] = (arr[i][j][k][l] - mean) / stdDev; // Normalize
//                         result[i][j][k][l] *= 10; // Apply controlled amplification
//                     }
//                 }
//             }
//         }
//         return result;
//     }

//     public double[][] rotate180(double[][] arr) {
//         int height = arr.length;
//         int width = arr[0].length;
//         double[][] result = new double[height][width];

//         for (int i = 0; i < height; i++) {
//             for (int j = 0; j < width; j++) {
//                 result[height - 1 - i][width - 1 - j] = arr[i][j];
//             }
//         }
//         return result;
//     }

//     // Update filters using momentum
//     private void updateFiltersWithMomentum(double[][][][] filters, double[][][][] dFilters, double[][][][] vFilters, double learningRate) {
//         ForkJoinPool pool = ForkJoinPool.commonPool();
    
//         pool.submit(() -> IntStream.range(0, filters.length).parallel().forEach(k -> {
//             for (int c = 0; c < filters[0].length; c++) {
//                 for (int i = 0; i < filters[0][0].length; i++) {
//                     for (int j = 0; j < filters[0][0][0].length; j++) {
//                         vFilters[k][c][i][j] = beta * vFilters[k][c][i][j] - learningRate * dFilters[k][c][i][j];
//                         filters[k][c][i][j] += vFilters[k][c][i][j];
//                     }
//                 }
//             }
//         })).join();
//     }

//     // Update biases using momentum
//     private void updateBiasesWithMomentum(double[] biases, double[] dBiases, double[] vBiases,double learningRate) {
//         for (int k = 0; k < numFilters; k++) {
//             vBiases[k] = beta * vBiases[k] + (1 - beta) * dBiases[k];
//             biases[k] -= learningRate * vBiases[k];
//         }
//     }
    
// }
