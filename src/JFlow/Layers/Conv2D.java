package JFlow.Layers;

import java.util.stream.IntStream;

import JFlow.Utility;

import java.util.*;
import java.util.concurrent.*;

public class Conv2D extends Layer {
    private double[] A, Z, dZ, filters, dFilters, vFilters, lastInput, dX;
    private double[] biases, dBiases, vBiases;
    private final double beta = 0.9; // Momentum coefficient
    private Activation activation;
    private int numFilters, filterSize, numChannels, inputHeight, inputWidth, numImages;
    private String padding;
    private static final ExecutorService threadPool = Executors.newFixedThreadPool(
        Runtime.getRuntime().availableProcessors()
    );

    protected Conv2D(int numFilters, int inputChannels, int filterSize, String padding) {
        super(numFilters * filterSize * filterSize, "conv_2d");
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.numChannels = inputChannels;
        this.padding = padding;

        // Initialize filters with He
        Random rand = new Random();
        double stdDev = Math.sqrt(2.0 / (numChannels * filterSize * filterSize));

        double filterScale = 1.0;

        int filterSizeTotal = numFilters * numChannels * filterSize * filterSize;
        filters = new double[filterSizeTotal];
        vFilters = new double[filterSizeTotal];

        for (int i = 0; i < filterSizeTotal; i++) {
            filters[i] = rand.nextGaussian() * stdDev * filterScale;
        }

        // Initialize biases to 0
        biases = new double[numFilters];
        vBiases = new double[numFilters];
    }


    @Override
    public void setActivation(Activation activation) {
        this.activation = activation;
    }

    @Override
    public void forward(double[] input, boolean training, int numImages, int channels, int height, int width) {
        lastInput = input;
        this.inputHeight = height;
        this.inputWidth = width;
        this.numImages = numImages;

        // System.out.println("input images, first 5 values: " + Arrays.toString(Arrays.copyOf(input, 28)));
        // System.out.println("Max input: " + Utility.max(input));

        int outputSize = numImages * numFilters * height * width;
        A = new double[outputSize];

        List<Callable<Void>> tasks = new ArrayList<>();

        for (int j = 0; j < numImages; j++) {
            for (int k = 0; k < numFilters; k++) {
                final int filterIndex = k;
                final int imageIndex = j;
                tasks.add(() -> {
                    int startIdx = imageIndex * numChannels * height * width;
                    int outputIdx = (imageIndex * numFilters + filterIndex) * height * width;
                    applyConv2D(A, outputIdx, input, startIdx, filters, filterIndex, biases[filterIndex]);
                    return null;
                });
            }
        }
        try {
            threadPool.invokeAll(tasks);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }


        if (super.getDropout() != null && training) {
            super.getDropout().newDropoutMaskConv(numFilters); // Generate new mask
            super.getDropout().applyDropoutConv(A, numImages, numFilters, height, width); // Apply dropout
        }



        // System.out.println("Before activation, first 5 values: " + Arrays.toString(Arrays.copyOf(A, 5)));

        Z = activation.applyActivation(A);


        if (getNextLayer() != null) {
            getNextLayer().forward(Z, training, numImages, numFilters, height, width);
        }
    }

    @Override
    public void forward(double[][] input, boolean training) {
        // TODO: Implement or remove this method if unnecessary
        throw new UnsupportedOperationException("Unimplemented method 'forward'");
    }

    @Override
    public void backward(double[] input, double learningRate, int numImages, int inputChannels, int height, int width) {

        try {
            dZ = activation.applyDActivation(input, Z);
        } catch (Exception e) {
            System.exit(0);
        }
        

        if (super.getDebug()) {
            System.out.println("Max dZ: " + Utility.max(dZ));
            System.out.println("Max lastInput: " + Utility.max(lastInput));
        }


        if (super.getDropout() != null) {
            super.getDropout().applyDropoutConv(dZ, numImages, numFilters, height, width);
        }


        dFilters = new double[filters.length];
        dBiases = new double[numFilters];

        ForkJoinPool pool = ForkJoinPool.commonPool();
        try {
            pool.submit(() -> IntStream.range(0, numFilters).parallel().forEach(k -> {
                for (int c = 0; c < numChannels; c++) {
                    double[] accumulatedSum = new double[filterSize * filterSize]; 
        
                    for (int i = 0; i < numImages; i++) {
                        int lastInputIdx = (i * numChannels + c) * inputHeight * inputWidth;
                        int dZIdx = (i * numFilters + k) * (inputHeight - filterSize + 1) * (inputWidth - filterSize + 1);
        
                        if (dZIdx >= dZ.length) {
                            throw new ArrayIndexOutOfBoundsException("dZIdx out of bounds: " + dZIdx);
                        }
        
                        applyConv2DGradient(accumulatedSum, lastInput, lastInputIdx, dZ, dZIdx);
                    }
        
                    int filterIndex = ((k * numChannels) + c) * filterSize * filterSize;
                    if (filterIndex + filterSize * filterSize > dFilters.length) {
                        throw new ArrayIndexOutOfBoundsException("dFilters overflow: " + (filterIndex + filterSize * filterSize));
                    }
        
                    // System.arraycopy(accumulatedSum, 0, dFilters, filterIndex, filterSize * filterSize);
                    synchronized (dFilters) {
                        System.arraycopy(accumulatedSum, 0, dFilters, filterIndex, filterSize * filterSize);
                    }
                }
            })).get();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

        dFilters = Utility.clip(dFilters, -1.0, 1.0);
        if (super.getDebug()) 
            System.out.println("Max dFilters: " + Utility.max(dFilters));


        // System.out.println("Conv2D Backprop: numChannels=" + numChannels + ", inputChannels=" + inputChannels);
        // System.out.println("Expected dX size: " + (numImages * numChannels * inputHeight * inputWidth));
        // System.out.println("Actual dX size: " + dX.length);

        // Calculate dBiases
        Arrays.fill(dBiases, 0);
        IntStream.range(0, numFilters).parallel().forEach(k -> {
            double sum = 0;
            for (int i = 0; i < numImages; i++) {
                int dZIdx = (i * numFilters + k) * inputHeight * inputWidth;
                for (int j = 0; j < inputHeight * inputWidth; j++) {
                    sum += dZ[dZIdx + j];
                }
            }
            // dBiases[k] = sum;
            synchronized (dBiases) {
                dBiases[k] += sum;
            }            
        });
        dBiases = Utility.clip(dBiases, -0.1, 0.1);

        if (super.getDebug())
            System.out.println("Max dBiases: " + Utility.max(dBiases));
        
        // System.exit(0);

        // Normalize for batch size
        for (int i = 0; i < dFilters.length; i++) {
            dFilters[i] /= numImages;
        }
        for (int k = 0; k < dBiases.length; k++) {
            dBiases[k] /= numImages;
        }

        updateFiltersWithMomentum(filters, dFilters, vFilters, learningRate);
        updateBiasesWithMomentum(biases, dBiases, vBiases, learningRate);

                    // Calculate dX
            dX = new double[numImages * numChannels * inputHeight * inputWidth];

            int outputHeight = inputHeight - filterSize + 1;
            int outputWidth = inputWidth - filterSize + 1;
    
            try {
                pool.submit(() -> IntStream.range(0, numImages).parallel().forEach(i -> {
                    for (int c = 0; c < numChannels; c++) {  // Fix: Loop should use numChannels
                        double[] accumulatedGradients = new double[inputHeight * inputWidth];
    
                        for (int k = 0; k < numFilters; k++) {
                            int filterIndex = (k * numChannels + c) * filterSize * filterSize;  // Fix: use numChannels
                            int dZIndex = (i * numFilters + k) * outputHeight * outputWidth;  // Fix: numFilters should be used instead of numChannels
                            
                            applyConv2DGradient(accumulatedGradients, filters, filterIndex, dZ, dZIndex);
                        }
    
                        int dXIdx = (i * numChannels + c) * inputHeight * inputWidth;  // Fix: use numChannels
                        System.arraycopy(accumulatedGradients, 0, dX, dXIdx, inputHeight * inputWidth);
                    }
                })).get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
            if (super.getDebug())
                System.out.println("Max dX: " + Utility.max(dX));
        if (getPreviousLayer() != null) {
            getPreviousLayer().backward(dX, learningRate, numImages, numChannels, height, width);
        }
    }

    @Override
    public void backward(double[][] input, double learningRate) {
        int batchSize = input.length;
        int featureSize = input[0].length; 
    
        double[] flattened = new double[batchSize * featureSize];
        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(input[i], 0, flattened, i * featureSize, featureSize);
        }

        // assume same padding for now!
            // if (padding.equals("same_padding")) {
            //     outputHeight = inputHeight;
            //     outputWidth = inputWidth;
            // } else {
            //     outputHeight = inputHeight - filterSize + 1;
            //     outputWidth = inputWidth - filterSize + 1;
            // }

    
        backward(flattened, learningRate, batchSize, numFilters, inputHeight, inputWidth);
    }
    

    @Override
    public double[][] getOutput() {
        int size = Z.length / numImages;
        double[][] reshaped = new double[numImages][size];
        for (int i = 0; i < numImages; i++) {
            for (int s = 0; s < size; s++) {
                reshaped[i][s] = Z[i * size + s];
            }
        }
        return reshaped;
    }

    private void applyConv2D(double[] output, int outIdx, double[] input, int inIdx, double[] filter, int filterIdx, double bias) {
        int pad = 0;
        int outputHeight, outputWidth;
    
        if (padding.equals("same_padding")) {
            pad = filterSize / 2;
            outputHeight = inputHeight;
            outputWidth = inputWidth;
            // outputHeight = (inputHeight + 2 * pad - filterSize) / 2;
            // outputWidth = (inputWidth + 2 * pad - filterSize) / 2; 
        } else { // "valid" or no padding
            outputHeight = inputHeight - filterSize + 1;
            outputWidth = inputWidth - filterSize + 1;
        }
    
        int finalPad = pad;
        int finalOutputWidth = outputWidth;
    
        ForkJoinPool pool = ForkJoinPool.commonPool();
        try {
            pool.submit(() -> IntStream.range(0, outputHeight).parallel().forEach(i -> {
                for (int j = 0; j < finalOutputWidth; j++) {
                    double sum = bias;
    
                    for (int c = 0; c < numChannels; c++) {
                        for (int fi = 0; fi < filterSize; fi++) {
                            for (int fj = 0; fj < filterSize; fj++) {
                                int inputRow = i + fi - finalPad;
                                int inputCol = j + fj - finalPad;
    
                                if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth) {
                                    int inputIdx = ((c * inputHeight + inputRow) * inputWidth) + inputCol;
                                    int filterIdxOffset = ((c * filterSize + fi) * filterSize) + fj;
                                    sum += input[inIdx + inputIdx] * filter[filterIdx + filterIdxOffset];
                                }
                            }
                        }
                    }
    
                    output[outIdx + (i * finalOutputWidth + j)] = sum;
                }
            })).get();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
    }

    // private void applyConv2DGradient(double[] gradient, double[] input, int inIdx, double[] dZ, int dZIdx) {
    //     Arrays.fill(gradient, 0);

    //     // System.out.println("DEBUG: Inside applyConv2DGradient");
    //     // System.out.println("  dZIndex: " + dZIdx);
    //     // System.out.println("  dZ.length: " + dZ.length);

    
    //     int outputHeight = inputHeight - filterSize + 1;
    //     int outputWidth = inputWidth - filterSize + 1;
    
    //     for (int fi = 0; fi < filterSize; fi++) {
    //         for (int fj = 0; fj < filterSize; fj++) {
    //             double sum = 0.0;
    
    //             for (int i = 0; i < outputHeight; i++) {
    //                 for (int j = 0; j < outputWidth; j++) {
    //                     int dZRow = i;
    //                     int dZCol = j;
    
    //                     // Compute the correct index inside dZ
    //                     int dZIdxOffset = (dZRow * outputWidth) + dZCol;
    
    //                     // Print debugging information
    //                     if (dZIdx + dZIdxOffset >= dZ.length || dZIdxOffset >= outputHeight * outputWidth) {
    //                         System.err.println("ERROR: dZ index out of bounds!");
    //                         System.err.println("  dZIdx: " + dZIdx);
    //                         System.err.println("  dZIdxOffset: " + dZIdxOffset);
    //                         System.err.println("  dZ.length: " + dZ.length);
    //                         throw new ArrayIndexOutOfBoundsException("dZ index out of bounds: " + (dZIdx + dZIdxOffset));
    //                     }
    
    //                     sum += dZ[dZIdx + dZIdxOffset]; 
    //                 }
    //             }
    
    //             gradient[fi * filterSize + fj] += sum;
    //         }
    //     }
    // }

    private void applyConv2DGradient(double[] gradient, double[] input, int inIdx, double[] dZ, int dZIdx) {
        Arrays.fill(gradient, 0);
    
        int outputHeight = inputHeight - filterSize + 1;
        int outputWidth = inputWidth - filterSize + 1;
    
        for (int fi = 0; fi < filterSize; fi++) {
            for (int fj = 0; fj < filterSize; fj++) {
                double sum = 0.0;
    
                for (int i = 0; i < outputHeight; i++) {
                    for (int j = 0; j < outputWidth; j++) {
                        int dZRow = i;
                        int dZCol = j;
                        int dZIdxOffset = dZRow * outputWidth + dZCol;
    
                        // Prevent out-of-bounds errors
                        if (dZIdx + dZIdxOffset >= dZ.length) {
                            continue;
                        }
    
                        // **FLIP the filter indices (180° rotation)**
                        int flippedFi = filterSize - 1 - fi;
                        int flippedFj = filterSize - 1 - fj;
    
                        // Accumulate gradient correctly
                        gradient[flippedFi * filterSize + flippedFj] += dZ[dZIdx + dZIdxOffset];
                    }
                }
            }
        }
    }
    
    private void updateFiltersWithMomentum(double[] filters, double[] dFilters, double[] vFilters, double learningRate) {
        IntStream.range(0, filters.length).parallel().forEach(i -> {
            vFilters[i] = beta * vFilters[i] + (1 - beta) * dFilters[i];
            filters[i] -= learningRate * vFilters[i];
            
        });
    }
    
    private void updateBiasesWithMomentum(double[] biases, double[] dBiases, double[] vBiases, double learningRate) {
        IntStream.range(0, biases.length).parallel().forEach(i -> {
            vBiases[i] = beta * vBiases[i] + (1 - beta) * dBiases[i];
            biases[i] -= learningRate * vBiases[i];
        });
    }

    public double[] getFilters() {
        return filters;
    }

    public double[] getBiases() {
        return biases;
    }


    @Override
    public double[][] getGradient() {
        int size = dX.length / numImages;
        double[][] reshaped = new double[numImages][size];
        for (int i = 0; i < numImages; i++) {
            for (int s = 0; s < size; s++) {
                reshaped[i][s] = dX[i * size + s];
            }
        }
        return reshaped;
    }
    
}

