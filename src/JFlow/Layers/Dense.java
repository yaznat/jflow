package JFlow.Layers;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

import JFlow.JMatrix;
import JFlow.Utility;

class Dense extends Layer{
    private JMatrix weights, dWeights, vWeights, A, Z, dZ, lastInput, gOutput, biases, dBiases, vBiases;
    // For momentum weight updates
    private final double beta = 0.9;


    protected Dense(int inputSize, int outputSize) {
        super(inputSize * outputSize + outputSize, "dense");
        // Initialize weights and biases
        double[] weights = new double[outputSize * inputSize];
        double[] biases = new double[outputSize];

        // Xavier
        // double scale = 1.0 / Math.sqrt(inputSize) * 0.2; 
        // He
        double scale = Math.sqrt(2.0 / inputSize);
        // double scale =1;

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i * inputSize + j] = (Math.random() - 0.5) * scale;  
            }
            biases[i] = (Math.random() - 0.5) * 0.5;
        }

        this.weights = new JMatrix(weights, outputSize, inputSize, 1, 1);
        this.biases = new JMatrix(outputSize, 1, 1, 1);

        this.dWeights = new JMatrix(outputSize, inputSize, 1, 1);
        this.dBiases = new JMatrix(outputSize, 1, 1, 1);

        // Initialize vWeights and vBiases to 0
        vWeights = new JMatrix(outputSize, inputSize, 1, 1);
        vBiases = new JMatrix(outputSize, 1, 1, 1);
    }

    public double[] getWeights() {
        return weights.getMatrix();
    }
    public double[] getBiases() {
        return biases.getMatrix();
    }

    @Override
    public void forward(JMatrix input, boolean training) {
        if (getPreviousLayer() == null || !(getPreviousLayer() instanceof Dense)) {
            input = input.transpose2D();
        }
        lastInput = input;

        try {
            A = weights.dot(input, true); // scaled dot product
        } catch (IllegalArgumentException e) {

            A = weights.transpose2D().dot(input, true);
        }


        applyBiasByRow(A, biases); 

        if (getActivation() != null) {
            Z = getActivation().applyActivation(A);
        } else {
            Z = A;
        }
        



        if (getDropout() != null && training) {
            getDropout().newDropoutMask(Z.length(), Z.channels() * Z.height() * Z.width()); // Generate new mask
            Z = getDropout().applyDropout(Z);
        }

        if (getNextLayer() != null) {
            getNextLayer().forward(Z, training);
        }
    }
    @Override
    public void backward(JMatrix gradient, double learningRate) {

        if (getDebug()) {
            System.out.println("Dense");
            System.out.println("Input images:" + gradient.length());
            System.out.println("Input channels:" + gradient.channels());
            System.out.println("Input height:" + gradient.height());
            System.out.println("Input width:" + gradient.width());
        }

        // Calculate dActivation
        if (getActivation() != null) {
            dZ = getActivation().applyDActivation(Z, gradient);
        } else {
            dZ = gradient;
        }
        if (super.getDropout() != null) {
            dZ = getDropout().applyDropout(dZ);
        }


        try {
            dWeights = dZ.dot(lastInput.transpose2D(), true);
            dBiases.setMatrix(dZ.sum0(true));
        } catch (IllegalArgumentException e) {
            dWeights = dZ.transpose2D().dot(lastInput.transpose2D(), true);
            dBiases.setMatrix(dZ.transpose2D().sum0(true));
        }

        int batchSize = lastInput.channels() * lastInput.height() * lastInput.width();
        dWeights = adaptiveGradientClip(weights, dWeights, 1e-2);
        dBiases = dBiases.multiply(1.0 / batchSize);
        // dWeights.clip(-1.0, 1.0);
        //dWeights = adaptiveGradientClip(weights, dWeights, 1e-2);

        // dWeights = clipGradients(dWeights, 3.0);
        // System.out.println(Utility.max(dWeights));
            


        

        if (getDebug()){
            System.out.println("Max dense weights: " + weights.max());
            System.out.println("Max dense biases: " + biases.max());
            System.out.println("Max dense dWeights: " + dWeights.max());
            System.out.println("Max dense dbiases: " + dBiases.max());
           }

        // for (int i = 0; i < dBiases.length; i++) {
        //     dBiases[i] *= (1.0 / numSamples);
        // }
        // dBiases = Utility.clip(dBiases, -0.2, 0.2);


        // Compute velocity update
        vWeights = vWeights.multiply(beta).add(dWeights.multiply(1 - beta));
        vBiases = vBiases.multiply(beta).add(dBiases.multiply(1 - beta));

        // System.out.println("MaxdDense: " + Utility.max(vWeights));
        // Apply gradients
        weights = weights.subtract(vWeights.multiply(learningRate));
        biases = biases.subtract(vBiases.multiply(learningRate));
        // System.out.println(Utility.max(weights));

        // Calculate loss w.r.t previous layer
        try {
            gOutput = weights.transpose2D().dot(dZ, true); // scaled
        } catch (IllegalArgumentException e) {
            gOutput = weights.transpose2D().dot(dZ.transpose2D(), true); // scaled
        }
         

        if (getPreviousLayer() != null) {
            getPreviousLayer().backward(gOutput, learningRate);
        }
    }

    @Override
    public JMatrix getOutput() {
        return Z;
    }


    private void applyBiasByRow(JMatrix A, JMatrix bias) {
        double[] aMatrix = A.getMatrix();
        double[] biasMatrix = bias.getMatrix();
        int rows = A.length();
        int cols = A.channels() * A.height() * A.width();
        for (int i = 0; i < rows; i++) { 
            for (int j = 0; j < cols; j++) { 
                aMatrix[i * cols + j] += biasMatrix[i]; 
            }
        }
    }
    
    private JMatrix adaptiveGradientClip(JMatrix weights, JMatrix biases, double epsilon) {
        double weightNorm = weights.frobeniusNorm();
        double gradNorm = dWeights.frobeniusNorm();
        double maxNorm = Math.max(gradNorm, epsilon * weightNorm);

        if (gradNorm > maxNorm) {
            double scale = maxNorm / gradNorm;
            return dWeights.multiply(scale);
        }
        return dWeights;
    }

    private double[][] adaptiveGradientClip(double[][] weights, double[][] dWeights, double epsilon) {
        double weightNorm = frobeniusNorm(weights);
        double gradNorm = frobeniusNorm(dWeights);
        double maxNorm = Math.max(gradNorm, epsilon * weightNorm);
        
        if (gradNorm > maxNorm) {
            double scale = maxNorm / gradNorm;
            return Utility.multiply(dWeights, scale);
        }
        return dWeights;
    }

    // // Calculate gradient norm
    // private double computeNorm(double[][] gradients) {
    //     double sum = 0.0;
    //     for (int i = 0; i < gradients.length; i++) {
    //         for (int j = 0; j < gradients[0].length; j++) {
    //             sum += gradients[i][j] * gradients[i][j]; 
    //         }
    //     }
    //     return Math.sqrt(sum); 
    // }

    // // Clip biases with norm consideration
    // private double[][] clipGradients(double[][] gradients, double clipNorm) {
    //     double norm = computeNorm(gradients);
        
    //     if (norm > clipNorm) {
    //         double scale = clipNorm / norm; 
    //         return Utility.multiply(gradients, scale); 
    //     }
        
    //     return gradients; 
    // }

    private double frobeniusNorm(double[][] matrix) {
        double sum = 0.0;
        for (double[] row : matrix) {
            for (double val : row) {
                sum += val * val;
            }
        }
        return Math.sqrt(sum);
    }
    
    // private double[][] matrixMultiply(double[][] arr1, double[][] arr2, boolean scale) {
    //     if (arr1[0].length != arr2.length) { 
    //         throw new IllegalArgumentException("Matrix multiplication not possible for arrays with shape: (" 
    //                 + arr1.length + "," + arr1[0].length + ") and (" + arr2.length + "," + arr2[0].length + ")");
    //     }

    //     int m = arr1.length;      
    //     int n = arr2[0].length;   
    //     int k = arr1[0].length;   

    //     double scaleFactor = scale ? 1.0 / Math.sqrt(k) : 1.0;
    //     double[][] result = new double[m][n];

    //     ForkJoinPool.commonPool().invoke(new MultiplyTask(arr1, arr2, result, 0, m, n, k, scaleFactor));

    //     return result;
    // }

    // ReLU activation function
    public static double[][] ReLU(double[][] A) {
        int numRows = A.length;
        int numCols = A[0].length;
    
        int numThreads = Runtime.getRuntime().availableProcessors();
        int minSizeForParallel = 10000 * numThreads;
        double[][] result = new double[numRows][numCols];
        
        if (numRows * numCols >= minSizeForParallel) {
            ForkJoinPool pool = ForkJoinPool.commonPool();
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int i = 0; i < numRows; i++) {
                final int row = i;
                tasks.add(() -> {
                    for (int col = 0; col < numCols; col++) {
                        result[row][col] = Math.max(A[row][col], 0);
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
        }  else {
            for (int row = 0; row < numRows; row++) {
                for (int col = 0; col < numCols; col++) {
                    result[row][col] = Math.max(A[row][col], 0);
                }
            }
        }
        return result;
    }
    // leaky ReLU activation function
    public static double[][] leakyReLU(double[][] A, double alpha) {
        int numRows = A.length;
        int numCols = A[0].length;
    
        int numThreads = Runtime.getRuntime().availableProcessors();
        int minSizeForParallel = 10000 * numThreads;
        double[][] result = new double[numRows][numCols];
        
        if (numRows * numCols >= minSizeForParallel) {
            ForkJoinPool pool = ForkJoinPool.commonPool();
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int i = 0; i < numRows; i++) {
                final int row = i;
                tasks.add(() -> {
                    for (int col = 0; col < numCols; col++) {
                        result[row][col] = Math.max(A[row][col], alpha);
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
        }  else {
            for (int row = 0; row < numRows; row++) {
                for (int col = 0; col < numCols; col++) {
                    result[row][col] = Math.max(A[row][col], alpha);
                }
            }
        }
        return result;
    }

    // derivative of ReLU activation function
    public static double[][] dReLU(double[][] arr) {
        int numRows = arr.length;
        int numCols = arr[0].length;

        int numThreads = Runtime.getRuntime().availableProcessors();
        int minSizeForParallel = 10000 * numThreads;
        double[][] result = new double[numRows][numCols];


        if (numRows * numCols >= minSizeForParallel) {
            ForkJoinPool pool = ForkJoinPool.commonPool();
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int i = 0; i < numRows; i++) {
                final int row = i;
                tasks.add(() -> {
                    for (int col = 0; col < numCols; col++) {
                        if (arr[row][col] > 0) {
                            result[row][col] = 1;
                        } else {
                            result[row][col] = 0;
                        }
                            
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
        }  else {
            for (int row = 0; row < numRows; row++) {
                for (int col = 0; col < numCols; col++) {
                    if (arr[row][col] > 0) {
                        result[row][col] = 1;
                    } else {
                        result[row][col] = 0;
                    }
                }
            }
        }
        return result;
    }
    // derivative of leakyReLU activation function
    public static double[][] dLeakyReLU(double[][] arr, double alpha) {
        int numRows = arr.length;
        int numCols = arr[0].length;

        int numThreads = Runtime.getRuntime().availableProcessors();
        int minSizeForParallel = 10000 * numThreads;
        double[][] result = new double[numRows][numCols];


        if (numRows * numCols >= minSizeForParallel) {
            ForkJoinPool pool = ForkJoinPool.commonPool();
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int i = 0; i < numRows; i++) {
                final int row = i;
                tasks.add(() -> {
                    for (int col = 0; col < numCols; col++) {
                        if (arr[row][col] > 0) {
                            result[row][col] = 1;
                        } else {
                            result[row][col] = alpha;
                        }
                            
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
        }  else {
            for (int row = 0; row < numRows; row++) {
                for (int col = 0; col < numCols; col++) {
                    if (arr[row][col] > 0) {
                        result[row][col] = 1;
                    } else {
                        result[row][col] = alpha;
                    }
                }
            }
        }
        return result;
    }

    // Softmax activation function
    public static double[][] softmax(double[][] arr, boolean transpose) {
        double[][] result = new double[arr.length][arr[0].length];
        
        if (transpose) {
            for (int y = 0; y < arr[0].length; y++) {
                double max = Double.NEGATIVE_INFINITY;
                
                // Find the max value in the column
                for (int x = 0; x < arr.length; x++) {
                    if (arr[x][y] > max) {
                        max = arr[x][y];
                    }
                }
                
                double sum = 0;
                
                // Compute the sum of exponentials after subtracting the max value for numerical stability
                for (int x = 0; x < arr.length; x++) {
                    sum += Math.exp(arr[x][y] - max);
                }
                
                // Compute the softmax for each element
                for (int x = 0; x < arr.length; x++) {
                    result[x][y] = Math.exp(arr[x][y] - max) / sum;
                }
            }
        } else {
    
            for (int i = 0; i < arr.length; i++) { 
                double maxVal = Double.NEGATIVE_INFINITY;
        
                for (int j = 0; j < arr[i].length; j++) {
                    if (arr[i][j] > maxVal) {
                        maxVal = arr[i][j];
                    }
                }
        
                double sumExp = 0.0;
                double[] expValues = new double[arr[i].length];
                for (int j = 0; j < arr[i].length; j++) {
                    expValues[j] = Math.exp(arr[i][j] - maxVal); 
                    sumExp += expValues[j];
                }
        
                for (int j = 0; j < arr[i].length; j++) {
                    result[i][j] = expValues[j] / sumExp;
                }
            }
        }
        
        return result;
    }

    // Sigmoid activation function
    public static double[][] sigmoid(double[][] arr, boolean transpose) {
        double[][] result = new double[arr.length][arr[0].length];

        if (transpose) {
            for (int y = 0; y < arr[0].length; y++) {
                for (int x = 0; x < arr.length; x++) {
                    result[x][y] = 1.0 / (1.0 + Math.exp(-arr[x][y]));
                }
            }
        } else {
            for (int i = 0; i < arr.length; i++) {
                for (int j = 0; j < arr[i].length; j++) {
                    result[i][j] = 1.0 / (1.0 + Math.exp(-arr[i][j]));
                }
            }
        }

        return result;
    }

    // Derivative of sigmoid activation function
    public static double[][] dSigmoid(double[][] A) {
        double[][] result = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[i].length; j++) {
                result[i][j] = A[i][j] * (1 - A[i][j]); 
            }
        }
        return result;
    }

    @Override
    public JMatrix getGradient() {
        return gOutput;
    }



}



