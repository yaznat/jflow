package JFlow.Layers;

import java.util.stream.IntStream;

import JFlow.JMatrix;
import JFlow.Utility;

import java.util.*;
import java.util.concurrent.*;

class Conv2D extends Layer {
    private JMatrix A, Z, dZ, filters, dFilters, vFilters, lastInput, dX;
    private JMatrix biases, dBiases, vBiases;
    private final double beta = 0.9; // Momentum coefficient
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
        double[] filters = new double[filterSizeTotal];

        for (int i = 0; i < filterSizeTotal; i++) {
            filters[i] = rand.nextGaussian() * stdDev * filterScale;
        }

        this.filters = new JMatrix(filters, numFilters, numChannels, filterSize, filterSize);

        // Initialize biases to 0
        biases = new JMatrix(numFilters, 1, 1, 1);

        dFilters = new JMatrix(numFilters, numChannels, filterSize, filterSize);
        dBiases = new JMatrix(numFilters, 1, 1, 1);

        vFilters = new JMatrix(numFilters, numChannels, filterSize, filterSize);
        vBiases = new JMatrix(numFilters, 1, 1, 1);
    }


    @Override
    public void forward(JMatrix input, boolean training) {
        lastInput = input;
        this.inputHeight = input.height();
        this.inputWidth = input.width();
        this.numImages = input.length();

        // System.out.println("input images, first 5 values: " + Arrays.toString(Arrays.copyOf(input, 28)));
        // System.out.println("Max input: " + Utility.max(input));

        int outputSize = numImages * numFilters * inputHeight * inputWidth;
        if (A == null || A.size() != outputSize) {  
            A = new JMatrix(numImages, numFilters, inputHeight, inputWidth);
        } else {
            A.fill(0);
        }

        List<Callable<Void>> tasks = new ArrayList<>();

        for (int j = 0; j < numImages; j++) {
            for (int k = 0; k < numFilters; k++) {
                final int filterIndex = k;
                final int imageIndex = j;
                tasks.add(() -> {
                    int startIdx = imageIndex * numChannels * inputHeight * inputWidth;
                    int outputIdx = (imageIndex * numFilters + filterIndex) * inputHeight * inputWidth;
                    applyConv2D(A.getMatrix(), outputIdx, input.getMatrix(), startIdx, 
                        filters.getMatrix(), filterIndex, biases.getMatrix()[filterIndex]);
                    return null;
                });
            }
        }
        try {
            threadPool.invokeAll(tasks);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }


        // if (super.getDropout() != null && training) {
        //     super.getDropout().newDropoutMaskConv(numFilters); // Generate new mask
        //     super.getDropout().applyDropoutConv(A, numImages, numFilters, height, width); // Apply dropout
        // }



        // System.out.println("Before activation, first 5 values: " + Arrays.toString(Arrays.copyOf(A, 5)));

        Z = getActivation().applyActivation(A);


        if (getNextLayer() != null) {
            getNextLayer().forward(Z, training);
        }
    }

    @Override
    public void backward(JMatrix input, double learningRate) {
        int height = input.height();
        int width = input.width();

        if (getActivation() != null) {
            dZ = getActivation().applyDActivation(Z, input);
        } else {
            dZ = input;
        }
        

        if (getDebug()) {
            System.out.println("Max dZ: " + dZ.max());
            System.out.println("Max lastInput: " + lastInput.max());
        }


        // if (super.getDropout() != null) {
        //     super.getDropout().applyDropoutConv(dZ, numImages, numFilters, height, width);
        // }



        dFilters.fill(0);

        double[] dZmatrix = dZ.getMatrix();
        double[] lastInputMatrix = lastInput.getMatrix();

        ForkJoinPool pool = ForkJoinPool.commonPool();
        try {
            pool.submit(() -> IntStream.range(0, numFilters).parallel().forEach(k -> {
                for (int c = 0; c < numChannels; c++) {
                    double[] accumulatedSum = new double[filterSize * filterSize]; 
        
                    for (int i = 0; i < numImages; i++) {
                        int lastInputIdx = (i * numChannels + c) * inputHeight * inputWidth;
                        int dZIdx = (i * numFilters + k) * (inputHeight - filterSize + 1) * (inputWidth - filterSize + 1);
        
                        if (dZIdx >= dZmatrix.length) {
                            throw new ArrayIndexOutOfBoundsException("dZIdx out of bounds: " + dZIdx);
                        }
        
                        applyConv2DGradient(accumulatedSum, lastInputMatrix, lastInputIdx, dZmatrix, dZIdx, false);
                    }
        
                    int filterIndex = ((k * numChannels) + c) * filterSize * filterSize;
                    if (filterIndex + filterSize * filterSize > dFilters.size()) {
                        throw new ArrayIndexOutOfBoundsException("dFilters overflow: " + (filterIndex + filterSize * filterSize));
                    }
        
                    // System.arraycopy(accumulatedSum, 0, dFilters, filterIndex, filterSize * filterSize);
                    System.arraycopy(accumulatedSum, 0, dFilters.getMatrix(), filterIndex, filterSize * filterSize);
                }
            })).get();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

        dFilters.clip(-1.0, 1.0);
        if (super.getDebug()) 
            System.out.println("Max dFilters: " + dFilters.max());


        // System.out.println("Conv2D Backprop: numChannels=" + numChannels + ", inputChannels=" + inputChannels);
        // System.out.println("Expected dX size: " + (numImages * numChannels * inputHeight * inputWidth));
        // System.out.println("Actual dX size: " + dX.length);


        // Calculate dBiases
        dBiases.fill(0);
        double[] dBiasesMatrix = dBiases.getMatrix();
        
        IntStream.range(0, numFilters).parallel().forEach(k -> {
            double sum = 0;
            for (int i = 0; i < numImages; i++) {
                int dZIdx = (i * numFilters + k) * inputHeight * inputWidth;
                for (int j = 0; j < inputHeight * inputWidth; j++) {
                    sum += dZmatrix[dZIdx + j];
                }
            }
            // dBiases[k] = sum;
            dBiasesMatrix[k] += sum;        
        });
        dBiases.clip(-0.1, 0.1);

        if (super.getDebug())
            System.out.println("Max dBiases: " + dBiases.max());
        
        // System.exit(0);

        // Normalize for batch size
        for (int i = 0; i < dFilters.size(); i++) {
            dFilters.getMatrix()[i] /= numImages;
        }
        for (int k = 0; k < dBiases.size(); k++) {
            dBiases.getMatrix()[k] /= numImages;
        }

        updateFiltersWithMomentum(filters.getMatrix(), dFilters.getMatrix(), vFilters.getMatrix(), learningRate);
        updateBiasesWithMomentum(biases.getMatrix(), dBiases.getMatrix(), vBiases.getMatrix(), learningRate);

        // Calculate dX
        if (dX == null || !dX.isSameShapeAs(lastInput)) {
            dX = new JMatrix(numImages, numChannels,  inputHeight, inputWidth);
        } else {
            dX.fill(0);
        }

            double[] dXmatrix = dX.getMatrix();
            double[] filtersMatrix = filters.getMatrix();

            int outputHeight = inputHeight - filterSize + 1;
            int outputWidth = inputWidth - filterSize + 1;
    
            try {
                pool.submit(() -> IntStream.range(0, numImages).parallel().forEach(i -> {
                    for (int c = 0; c < numChannels; c++) {  
                        double[] accumulatedGradients = new double[inputHeight * inputWidth];
    
                        for (int k = 0; k < numFilters; k++) {
                            int filterIndex = (k * numChannels + c) * filterSize * filterSize;  
                            int dZIndex = (i * numFilters + k) * outputHeight * outputWidth;  
                            
                            applyConv2DGradient(accumulatedGradients, filtersMatrix, filterIndex, dZmatrix, dZIndex, true);
                        }
    
                        int dXIdx = (i * numChannels + c) * inputHeight * inputWidth;  
                        System.arraycopy(accumulatedGradients, 0, dXmatrix, dXIdx, inputHeight * inputWidth);
                    }
                })).get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
            dX.clip(-5.0, 5.0);
            if (super.getDebug())
                System.out.println("Max dX: " + dX.max());
        if (getPreviousLayer() != null) {
            getPreviousLayer().backward(dX, learningRate);
        }
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

    private void applyConv2DGradient(double[] gradient, double[] input, int inIdx, double[] dZ, int dZIdx, boolean flipFilter) {
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
    
                        if (dZIdx + dZIdxOffset >= dZ.length) {
                            continue;
                        }
    
                        int filterFi = fi;
                        int filterFj = fj;
    
                        // Flip the filter only when calculating dX
                        if (flipFilter) {
                            filterFi = filterSize - 1 - fi;
                            filterFj = filterSize - 1 - fj;
                        }
    
                        // Accumulate gradient
                        gradient[filterFi * filterSize + filterFj] += dZ[dZIdx + dZIdxOffset];
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
        return filters.getMatrix();
    }

    public double[] getBiases() {
        return biases.getMatrix();
    }


    @Override
    public JMatrix getOutput() {
       return Z;
    }


    @Override
    public JMatrix getGradient() {
        return dX;
    }
    
}

