// package JFlow.Layers;

// import java.util.ArrayList;
// import java.util.List;
// import java.util.concurrent.Callable;
// import java.util.concurrent.ForkJoinPool;
// import java.util.stream.IntStream;

// import JFlow.Utility;

// class Dense extends Layer{
//     private double[][] weights, dWeights, vWeights, A, Z, dZ, lastInput;
//     private double[] biases, dBiases, vBiases;
//     private Dropout dropout = null;
//     // For momentum weight updates
//     private final double beta = 0.9;


//     protected Dense(int inputSize, int outputSize) {
//         super(inputSize * outputSize + inputSize, "dense");
//         // Initialize weights and biases
//         weights = new double[outputSize][inputSize];
//         biases = new double[outputSize];

//         // Xavier
//         // double scale = 1.0 / Math.sqrt(inputSize) * 0.2; 
//         // He
//         double scale = Math.sqrt(2.0 / inputSize);
//         for (int i = 0; i < outputSize; i++) {
//             for (int j = 0; j < inputSize; j++) {
//                 weights[i][j] = (Math.random() - 0.5) * scale;  
//             }
//             biases[i] = (Math.random() - 0.5) * 0.5;
//         }

//         // Initialize vWeights and vBiases to 0
//         vWeights = new double[outputSize][inputSize];
//         vBiases = new double[outputSize];
//     }

//     public double[][] getWeights() {
//         return weights;
//     }
//     public double[] getBiases() {
//         return biases;
//     }
//     public void setDropout(Dropout dropout) {
//         this.dropout = dropout;
//     }

//     @Override
//     public void forward(double[][][][] input, boolean training) {
//         forward(Utility.transpose(flatten(input)), training);
//     }

//     @Override
//     public void forward(double[][] input, boolean training) {
//         lastInput = input;

//         // System.out.println(input.length + "," + input[0].length);
//         try {
//             A = matrixMultiply(weights, input, true); // scaled dot product
//         } catch (IllegalArgumentException e) {
//             A = matrixMultiply(Utility.transpose(weights), input, true);
//         }


//         applyBiasByRow(A, biases); 



//         if (getActivation() instanceof ReLU) {
//             Z = ReLU(A);
//         } else if (getActivation() instanceof LeakyReLU) {
//             Z = leakyReLU(A, ((LeakyReLU)getActivation()).getAlpha());
//         } else if (getActivation() instanceof Softmax) {
//             Z = softmax(A, true);
//         } 

//         if (dropout != null && training) {
//             dropout.newDropoutMask(Z.length, Z[0].length); // Generate new mask
//             Z = dropout.applyDropout(Z);
//         }

//         if (getNextLayer() != null) {
//             getNextLayer().forward(Z, training);
//         }
//     }

//     @Override
//     public void backward(double[][][][] input, double learningRate) {
//         backward(flatten(input), learningRate);
//     }

//     @Override
//     public void backward(double[][] input, double learningRate) {
        
//         int numSamples = input[0].length; // transposed
//         // Calculate dActivation
//         if (getActivation() instanceof Softmax) {
//             dZ = Utility.subtract(Z, input);
//         } else if (getActivation() instanceof ReLU) {
//             dZ = Utility.multiply(dReLU(A), input);
//         } else if (getActivation() instanceof LeakyReLU) {
//             dZ = Utility.multiply(dLeakyReLU(A, ((LeakyReLU)getActivation()).getAlpha()), input);
//         }

//         if (dropout != null) {
//             dZ = dropout.applyDropoutDerivative(dZ);
//         }

//         // Calculate and apply dWeights and dBiases
//         // dWeights = Utility.multiply(
//         //     matrixMultiply(dZ, Utility.transpose(lastInput), true), 
//         //     (1.0 / numSamples));

//         dWeights = matrixMultiply(dZ, Utility.transpose(lastInput), true);

//         // dWeights = Utility.clip(dWeights, -1.0, 1.0);
//        dWeights = adaptiveGradientClip(weights, dWeights, 1e-2);

//         // dWeights = clipGradients(dWeights, 3.0);
//         // System.out.println(Utility.max(dWeights));
            


//         dBiases = Utility.sum0(dZ, true); // scaled

//         // for (int i = 0; i < dBiases.length; i++) {
//         //     dBiases[i] *= (1.0 / numSamples);
//         // }
//         // dBiases = Utility.clip(dBiases, -0.2, 0.2);

//         // System.out.println(Utility.max(new double[][]{dBiases}));

//         // Compute velocity update
//         vWeights = Utility.add(Utility.multiply(vWeights, beta), Utility.multiply(dWeights, (1 - beta)));
//         vBiases = Utility.add(Utility.multiply(vBiases, beta), Utility.multiply(dBiases, (1 - beta)));

//         // System.out.println("MaxdDense: " + Utility.max(vWeights));
//         // Apply gradients
//         weights = Utility.subtract(weights, Utility.multiply(vWeights, learningRate));
//         biases = Utility.subtract(biases, Utility.multiply(vBiases, learningRate));
//         // System.out.println(Utility.max(weights));

//         // Calculate loss w.r.t previous layer
//         double[][] output = matrixMultiply(Utility.transpose(weights), dZ, true); // scaled

//         if (getPreviousLayer() != null) {
//             getPreviousLayer().backward(output, learningRate * 0.85);
//         }
//     }

//     @Override
//     public double[][] getOutput() {
//         return Z;
//     }


//     private void applyBiasByRow(double[][] A, double[] bias) {
//         for (int i = 0; i < A.length; i++) { 
//             for (int j = 0; j < A[0].length; j++) { 
//                 A[i][j] += bias[i]; 
//             }
//         }
//     }
    



//     // Get the dot product of two arrays
//     private double[][] matrixMultiply(double[][] arr1, double[][] arr2, boolean scale) {
        
//         if (arr1[0].length != arr2.length) {
//             String errorMessage = "Matrix multiplication not possible for arrays with shape: (" 
//                                   + arr1.length + "," + arr1[0].length + ") and (" + arr2.length +
//                                   "," + arr2[0].length + ")";
//             throw new IllegalArgumentException(errorMessage);
//         }

//         double scaleFactor = 1.0 / Math.sqrt(arr1[0].length);

//         int m = arr1.length; // Number of rows in arr1
//         int n = arr2[0].length; // Number of columns in arr2
//         int k = arr1[0].length; // Number of columns in arr1 (same as number of rows in arr2)

//         double[][] result = new double[m][n];

//         ForkJoinPool pool = new ForkJoinPool(Runtime.getRuntime().availableProcessors());

//         pool.submit(() -> IntStream.range(0, m).parallel().forEach(i -> {
//             for (int j = 0; j < n; j++) {
//                 double sum = 0;
//                 for (int kIndex = 0; kIndex < k; kIndex++) {
//                     sum += arr1[i][kIndex] * arr2[kIndex][j];
//                 }
//                 if (scale) {
//                     result[i][j] = sum * scaleFactor;
//                 } else {
//                     result[i][j] = sum;
//                 }
//             }
//         })).join();

//         pool.shutdown();
//         pool.close();
        
//         return result;
//     }


//     private double[][] adaptiveGradientClip(double[][] weights, double[][] dWeights, double epsilon) {
//         double weightNorm = frobeniusNorm(weights);
//         double gradNorm = frobeniusNorm(dWeights);
//         double maxNorm = Math.max(gradNorm, epsilon * weightNorm);
        
//         if (gradNorm > maxNorm) {
//             double scale = maxNorm / gradNorm;
//             return Utility.multiply(dWeights, scale);
//         }
//         return dWeights;
//     }

//     // Calculate gradient norm
//     private double computeNorm(double[][] gradients) {
//         double sum = 0.0;
//         for (int i = 0; i < gradients.length; i++) {
//             for (int j = 0; j < gradients[0].length; j++) {
//                 sum += gradients[i][j] * gradients[i][j]; 
//             }
//         }
//         return Math.sqrt(sum); 
//     }

//     // Clip biases with norm consideration
//     private double[][] clipGradients(double[][] gradients, double clipNorm) {
//         double norm = computeNorm(gradients);
        
//         if (norm > clipNorm) {
//             double scale = clipNorm / norm; 
//             return Utility.multiply(gradients, scale); 
//         }
        
//         return gradients; 
//     }

//     private double frobeniusNorm(double[][] matrix) {
//         double sum = 0.0;
//         for (double[] row : matrix) {
//             for (double val : row) {
//                 sum += val * val;
//             }
//         }
//         return Math.sqrt(sum);
//     }
    
//     // private double[][] matrixMultiply(double[][] arr1, double[][] arr2, boolean scale) {
//     //     if (arr1[0].length != arr2.length) { 
//     //         throw new IllegalArgumentException("Matrix multiplication not possible for arrays with shape: (" 
//     //                 + arr1.length + "," + arr1[0].length + ") and (" + arr2.length + "," + arr2[0].length + ")");
//     //     }

//     //     int m = arr1.length;      
//     //     int n = arr2[0].length;   
//     //     int k = arr1[0].length;   

//     //     double scaleFactor = scale ? 1.0 / Math.sqrt(k) : 1.0;
//     //     double[][] result = new double[m][n];

//     //     ForkJoinPool.commonPool().invoke(new MultiplyTask(arr1, arr2, result, 0, m, n, k, scaleFactor));

//     //     return result;
//     // }

//     // ReLU activation function
//     public static double[][] ReLU(double[][] A) {
//         int numRows = A.length;
//         int numCols = A[0].length;
    
//         int numThreads = Runtime.getRuntime().availableProcessors();
//         int minSizeForParallel = 10000 * numThreads;
//         double[][] result = new double[numRows][numCols];
        
//         if (numRows * numCols >= minSizeForParallel) {
//             ForkJoinPool pool = ForkJoinPool.commonPool();
//             List<Callable<Void>> tasks = new ArrayList<>();
//             for (int i = 0; i < numRows; i++) {
//                 final int row = i;
//                 tasks.add(() -> {
//                     for (int col = 0; col < numCols; col++) {
//                         result[row][col] = Math.max(A[row][col], 0);
//                     }
//                     return null;
//                 });
//             }
//             pool.invokeAll(tasks);
//         }  else {
//             for (int row = 0; row < numRows; row++) {
//                 for (int col = 0; col < numCols; col++) {
//                     result[row][col] = Math.max(A[row][col], 0);
//                 }
//             }
//         }
//         return result;
//     }
//     // leaky ReLU activation function
//     public static double[][] leakyReLU(double[][] A, double alpha) {
//         int numRows = A.length;
//         int numCols = A[0].length;
    
//         int numThreads = Runtime.getRuntime().availableProcessors();
//         int minSizeForParallel = 10000 * numThreads;
//         double[][] result = new double[numRows][numCols];
        
//         if (numRows * numCols >= minSizeForParallel) {
//             ForkJoinPool pool = ForkJoinPool.commonPool();
//             List<Callable<Void>> tasks = new ArrayList<>();
//             for (int i = 0; i < numRows; i++) {
//                 final int row = i;
//                 tasks.add(() -> {
//                     for (int col = 0; col < numCols; col++) {
//                         result[row][col] = Math.max(A[row][col], alpha);
//                     }
//                     return null;
//                 });
//             }
//             pool.invokeAll(tasks);
//         }  else {
//             for (int row = 0; row < numRows; row++) {
//                 for (int col = 0; col < numCols; col++) {
//                     result[row][col] = Math.max(A[row][col], alpha);
//                 }
//             }
//         }
//         return result;
//     }

//     // derivative of ReLU activation function
//     public static double[][] dReLU(double[][] arr) {
//         int numRows = arr.length;
//         int numCols = arr[0].length;

//         int numThreads = Runtime.getRuntime().availableProcessors();
//         int minSizeForParallel = 10000 * numThreads;
//         double[][] result = new double[numRows][numCols];


//         if (numRows * numCols >= minSizeForParallel) {
//             ForkJoinPool pool = ForkJoinPool.commonPool();
//             List<Callable<Void>> tasks = new ArrayList<>();
//             for (int i = 0; i < numRows; i++) {
//                 final int row = i;
//                 tasks.add(() -> {
//                     for (int col = 0; col < numCols; col++) {
//                         if (arr[row][col] > 0) {
//                             result[row][col] = 1;
//                         } else {
//                             result[row][col] = 0;
//                         }
                            
//                     }
//                     return null;
//                 });
//             }
//             pool.invokeAll(tasks);
//         }  else {
//             for (int row = 0; row < numRows; row++) {
//                 for (int col = 0; col < numCols; col++) {
//                     if (arr[row][col] > 0) {
//                         result[row][col] = 1;
//                     } else {
//                         result[row][col] = 0;
//                     }
//                 }
//             }
//         }
//         return result;
//     }
//     // derivative of ReLU activation function
//     public static double[][] dLeakyReLU(double[][] arr, double alpha) {
//         int numRows = arr.length;
//         int numCols = arr[0].length;

//         int numThreads = Runtime.getRuntime().availableProcessors();
//         int minSizeForParallel = 10000 * numThreads;
//         double[][] result = new double[numRows][numCols];


//         if (numRows * numCols >= minSizeForParallel) {
//             ForkJoinPool pool = ForkJoinPool.commonPool();
//             List<Callable<Void>> tasks = new ArrayList<>();
//             for (int i = 0; i < numRows; i++) {
//                 final int row = i;
//                 tasks.add(() -> {
//                     for (int col = 0; col < numCols; col++) {
//                         if (arr[row][col] > 0) {
//                             result[row][col] = 1;
//                         } else {
//                             result[row][col] = alpha;
//                         }
                            
//                     }
//                     return null;
//                 });
//             }
//             pool.invokeAll(tasks);
//         }  else {
//             for (int row = 0; row < numRows; row++) {
//                 for (int col = 0; col < numCols; col++) {
//                     if (arr[row][col] > 0) {
//                         result[row][col] = 1;
//                     } else {
//                         result[row][col] = alpha;
//                     }
//                 }
//             }
//         }
//         return result;
//     }

//     // Softmax activation function
//     public static double[][] softmax(double[][] arr, boolean transpose) {
//         double[][] result = new double[arr.length][arr[0].length];
        
//         if (transpose) {
//             for (int y = 0; y < arr[0].length; y++) {
//                 double max = Double.NEGATIVE_INFINITY;
                
//                 // Find the max value in the column
//                 for (int x = 0; x < arr.length; x++) {
//                     if (arr[x][y] > max) {
//                         max = arr[x][y];
//                     }
//                 }
                
//                 double sum = 0;
                
//                 // Compute the sum of exponentials after subtracting the max value for numerical stability
//                 for (int x = 0; x < arr.length; x++) {
//                     sum += Math.exp(arr[x][y] - max);
//                 }
                
//                 // Compute the softmax for each element
//                 for (int x = 0; x < arr.length; x++) {
//                     result[x][y] = Math.exp(arr[x][y] - max) / sum;
//                 }
//             }
//         } else {
    
//             for (int i = 0; i < arr.length; i++) { 
//                 double maxVal = Double.NEGATIVE_INFINITY;
        
//                 for (int j = 0; j < arr[i].length; j++) {
//                     if (arr[i][j] > maxVal) {
//                         maxVal = arr[i][j];
//                     }
//                 }
        
//                 double sumExp = 0.0;
//                 double[] expValues = new double[arr[i].length];
//                 for (int j = 0; j < arr[i].length; j++) {
//                     expValues[j] = Math.exp(arr[i][j] - maxVal); 
//                     sumExp += expValues[j];
//                 }
        
//                 for (int j = 0; j < arr[i].length; j++) {
//                     result[i][j] = expValues[j] / sumExp;
//                 }
//             }
//         }
        
//         return result;
//     }


// }



