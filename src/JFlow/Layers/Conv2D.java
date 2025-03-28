package JFlow.Layers;

import java.util.stream.IntStream;

import JFlow.JMatrix;

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

        // Initialize vWeights and vBiases to 0
        vFilters = new JMatrix(numFilters, numChannels, filterSize, filterSize);
        vBiases = new JMatrix(numFilters, 1, 1, 1);
    }


    @Override
    public void forward(JMatrix input, boolean training) {
        lastInput = input;
        this.inputHeight = input.height();
        this.inputWidth = input.width();
        this.numImages = input.length();


        int outputSize = numImages * numFilters * inputHeight * inputWidth;
        if (A == null || A.size() != outputSize) {  
            A = new JMatrix(numImages, numFilters, inputHeight, inputWidth);
        } else {
            A.fill(0);
        }

        // Calculate forward output

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

    // Implement conv dropout eventually!
        // if (super.getDropout() != null && training) {
        //     super.getDropout().newDropoutMaskConv(numFilters); // Generate new mask
        //     super.getDropout().applyDropoutConv(A, numImages, numFilters, height, width); // Apply dropout
        // }

        // Apply activation
        if (getActivation() != null) {
            Z = getActivation().applyActivation(A);
        } else {
            Z = A;
        }
       
        if (getNextLayer() != null) {
            getNextLayer().forward(Z, training);
        }
    }

    @Override
    public void backward(JMatrix input, double learningRate) {
        // Apply activation derivative
        if (getActivation() != null) {
            dZ = getActivation().applyDActivation(Z, input);
        } else {
            dZ = input;
        }
        
        if (getDebug()) {
            System.out.println("Max dZ: " + dZ.max());
            System.out.println("Max lastInput: " + lastInput.max());
        }

    // Implement conv dropout eventually!
        // if (super.getDropout() != null) {
        //     super.getDropout().applyDropoutConv(dZ, numImages, numFilters, height, width);
        // }


        // Calculate dFilters
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
        
                    System.arraycopy(accumulatedSum, 0, dFilters.getMatrix(), filterIndex, filterSize * filterSize);
                }
            })).get();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

        // Clip values
        dFilters.clip(-1.0, 1.0);
        if (super.getDebug()) 
            System.out.println("Max dFilters: " + dFilters.max());



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
            dBiasesMatrix[k] += sum;        
        });
        // Clip values
        dBiases.clip(-0.1, 0.1);

        if (super.getDebug())
            System.out.println("Max dBiases: " + dBiases.max());
        

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
        // Clip values
        dX.clip(-5.0, 5.0);
        if (super.getDebug())
            System.out.println("Max dX: " + dX.max());

        if (getPreviousLayer() != null) {
            getPreviousLayer().backward(dX, learningRate);
        }
    }


    // Apply convolution to one image at a time
    private void applyConv2D(double[] output, int outIdx, double[] input, int inIdx, double[] filter, int filterIdx, double bias) {
        int pad = 0;
        int outputHeight, outputWidth;
    
        if (padding.equals("same_padding")) {
            pad = filterSize / 2;
            outputHeight = inputHeight;
            outputWidth = inputWidth;
        } else { // valid, or no, padding
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


    // Calculate the conv gradient for one image at a time
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

