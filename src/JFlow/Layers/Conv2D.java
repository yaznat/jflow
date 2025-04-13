package JFlow.Layers;

import java.util.stream.IntStream;

import JFlow.JMatrix;

import java.util.HashMap;
import java.util.Random;

class Conv2D extends Layer {
    private JMatrix A, Z, dZ, filters, dFilters, vFilters, lastInput, dX, biases, dBiases, vBiases;
    private final double beta = 0.9; // Momentum coefficient
    private int numFilters, filterSize, numChannels, inputHeight, inputWidth, numImages;
    private String padding;

    private double filterGradientClipNorm = 2.0; 
    private double biasGradientClipNorm = 1.0;   
    private double inputGradientClipNorm = 10.0;

    protected Conv2D(int numFilters, int filterSize, String padding) {
        super("conv_2d", 0);
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.padding = padding;
    }

    public void build(int inputChannels) {
        this.numChannels = inputChannels;
        setNumTrainableParameters(numFilters * inputChannels * filterSize * filterSize + numFilters);

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
    protected int channels() {
        return numFilters;
    }

    @Override
    protected void forward(JMatrix input, boolean training) {
        lastInput = input;
        this.inputHeight = input.height();
        this.inputWidth = input.width();
        this.numImages = input.length();
    
        // Calculate output dimensions based on padding
        int outputHeight, outputWidth;
        if (padding.equals("same_padding")) {
            outputHeight = inputHeight;
            outputWidth = inputWidth;
        } else { // valid padding
            outputHeight = inputHeight - filterSize + 1;
            outputWidth = inputWidth - filterSize + 1;
        }
    
        // Initialize output matrix with proper dimensions
        int outputSize = numImages * numFilters * outputHeight * outputWidth;
        if (A == null || A.size() != outputSize) {  
            A = new JMatrix(numImages, numFilters, outputHeight, outputWidth);
        } else {
            A.fill(0);
        }
    
        // Calculate forward output
        IntStream.range(0, numImages).parallel().forEach(j -> {
            for (int k = 0; k < numFilters; k++) {
                final int filterIndex = k;
                final int imageIndex = j;
                int startIdx = imageIndex * numChannels * inputHeight * inputWidth;
                int outputIdx = (imageIndex * numFilters + filterIndex) * outputHeight * outputWidth;
                applyConv2D(A.getMatrix(), outputIdx, input.getMatrix(), startIdx, 
                    filters.getMatrix(), filterIndex, biases.getMatrix()[filterIndex]);
            }
        });
        // Apply BatchNorm
        if (getBatchNorm() != null) {
            A = getBatchNorm().forward(A, training);
        }
    
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
    protected void backward(JMatrix input, double learningRate) {
        // Calculate output dimensions based on padding
        int outputHeight, outputWidth;
        if (padding.equals("same_padding")) {
            outputHeight = inputHeight;
            outputWidth = inputWidth;
        } else { // valid padding
            outputHeight = inputHeight - filterSize + 1;
            outputWidth = inputWidth - filterSize + 1;
        }

        // Apply activation derivative
        if (getActivation() != null) {
            dZ = getActivation().applyDActivation(Z, input);
        } else {
            dZ = input;
        }

         // Apply BatchNorm
         if (getBatchNorm() != null) {
            dZ = getBatchNorm().backward(dZ, learningRate);
        }
        
        if (getDebug()) {
            System.out.println("Max dZ: " + dZ.max());
            System.out.println("Max lastInput: " + lastInput.max());
        }

        // Calculate dFilters
        dFilters.fill(0);
        double[] dZmatrix = dZ.getMatrix();
        double[] lastInputMatrix = lastInput.getMatrix();

        // Calculate filter gradients for each filter and channel
        IntStream.range(0, numFilters).parallel().forEach(k -> {
            for (int c = 0; c < numChannels; c++) {
                double[] filterGradients = new double[filterSize * filterSize]; 

                for (int i = 0; i < numImages; i++) {
                    int lastInputIdx = (i * numChannels + c) * inputHeight * inputWidth;
                    int dZIdx = (i * numFilters + k) * outputHeight * outputWidth;

                    // Calculate correlation between input and output gradients
                    for (int fh = 0; fh < filterSize; fh++) {
                        for (int fw = 0; fw < filterSize; fw++) {
                            double sum = 0;
                            
                            for (int oh = 0; oh < outputHeight; oh++) {
                                for (int ow = 0; ow < outputWidth; ow++) {
                                    int dZPos = dZIdx + (oh * outputWidth + ow);
                                    
                                    // Calculate corresponding input position
                                    int inH, inW;
                                    
                                    if (padding.equals("same_padding")) {
                                        inH = oh + fh - filterSize / 2;
                                        inW = ow + fw - filterSize / 2;
                                    } else {
                                        inH = oh + fh;
                                        inW = ow + fw;
                                    }
                                    
                                    if (inH >= 0 && inH < inputHeight && inW >= 0 && inW < inputWidth &&
                                        dZPos >= 0 && dZPos < dZmatrix.length) {
                                        int inputPos = lastInputIdx + (inH * inputWidth + inW);
                                        
                                        if (inputPos >= 0 && inputPos < lastInputMatrix.length) {
                                            sum += lastInputMatrix[inputPos] * dZmatrix[dZPos];
                                        }
                                    }
                                }
                            }
                            
                            filterGradients[fh * filterSize + fw] += sum;
                        }
                    }
                }

                // Copy accumulated gradients to dFilters
                int filterGradientOffset = ((k * numChannels) + c) * filterSize * filterSize;
                System.arraycopy(filterGradients, 0, dFilters.getMatrix(), filterGradientOffset, filterSize * filterSize);
            }
        });

        // Scale values to prevent exploding gradients
        scaleFilterGradients();

        
        if (super.getDebug()) 
            System.out.println("Max dFilters: " + dFilters.max());

        // Calculate dBiases
        dBiases.fill(0);
        double[] dBiasesMatrix = dBiases.getMatrix();
        
        IntStream.range(0, numFilters).parallel().forEach(k -> {
            double sum = 0;
            for (int i = 0; i < numImages; i++) {
                int dZIdx = (i * numFilters + k) * outputHeight * outputWidth;
                for (int j = 0; j < outputHeight * outputWidth; j++) {
                    sum += dZmatrix[dZIdx + j];
                }
            }
            dBiasesMatrix[k] += sum;        
        });
        
        // Scale bias gradients
        scaleBiasGradients();


        if (super.getDebug())
            System.out.println("Max dBiases: " + dBiases.max());

        // Normalize for batch size
        for (int i = 0; i < dFilters.size(); i++) {
            dFilters.getMatrix()[i] /= numImages;
        }
        // for (int k = 0; k < dBiases.size(); k++) {
        //     dBiases.getMatrix()[k] /= numImages;
        // }

        updateFiltersWithMomentum(filters.getMatrix(), dFilters.getMatrix(), vFilters.getMatrix(), learningRate);
        updateBiasesWithMomentum(biases.getMatrix(), dBiases.getMatrix(), vBiases.getMatrix(), learningRate);

        // Calculate dX (input gradients)
        if (dX == null || !dX.isSameShapeAs(lastInput)) {
            dX = new JMatrix(numImages, numChannels, inputHeight, inputWidth);
        } else {
            dX.fill(0);
        }

        double[] dXmatrix = dX.getMatrix();
        double[] filtersMatrix = filters.getMatrix();

        IntStream.range(0, numImages).parallel().forEach(i -> {
            for (int c = 0; c < numChannels; c++) {  
                int dXBase = (i * numChannels + c) * inputHeight * inputWidth;
                
                for (int h = 0; h < inputHeight; h++) {
                    for (int w = 0; w < inputWidth; w++) {
                        double sum = 0;
                        
                        // For each filter
                        for (int k = 0; k < numFilters; k++) {
                            int filterBase = (k * numChannels + c) * filterSize * filterSize;
                            int dZBase = (i * numFilters + k) * outputHeight * outputWidth;
                            
                            // For each position in the filter
                            for (int fh = 0; fh < filterSize; fh++) {
                                for (int fw = 0; fw < filterSize; fw++) {
                                    // Flip filter by 180 degrees for gradient calculation 
                                    int flippedFh = filterSize - 1 - fh;
                                    int flippedFw = filterSize - 1 - fw;
                                    
                                    // Calculate corresponding output position
                                    int outH, outW;
                                    
                                    if (padding.equals("same_padding")) {
                                        outH = h - fh + filterSize / 2;
                                        outW = w - fw + filterSize / 2;
                                    } else {
                                        outH = h - fh;
                                        outW = w - fw;
                                    }
                                    
                                    if (outH >= 0 && outH < outputHeight && outW >= 0 && outW < outputWidth) {
                                        // Get filter value and output gradient
                                        int filterIdx = filterBase + (flippedFh * filterSize + flippedFw);
                                        int dZPos = dZBase + (outH * outputWidth + outW);
                                        
                                        if (dZPos >= 0 && dZPos < dZmatrix.length && 
                                            filterIdx >= 0 && filterIdx < filtersMatrix.length) {
                                            sum += filtersMatrix[filterIdx] * dZmatrix[dZPos];
                                        }
                                    }
                                }
                            }
                        }
                        
                        int dXPos = dXBase + (h * inputWidth + w);
                        dXmatrix[dXPos] = sum;
                    }
                }
            }
        });
        
        // Clip input gradients
        scaleInputGradients();
        
        
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
        
        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < finalOutputWidth; j++) {
                double sum = bias;
    
                for (int c = 0; c < numChannels; c++) {
                    for (int fi = 0; fi < filterSize; fi++) {
                        for (int fj = 0; fj < filterSize; fj++) {
                            int inputRow = i + fi - finalPad;
                            int inputCol = j + fj - finalPad;
    
                            if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth) {
                                int inputIdx = inIdx + (c * inputHeight * inputWidth) + (inputRow * inputWidth) + inputCol;
                                
                                int filterIdxOffset = (c * filterSize * filterSize) + (fi * filterSize) + fj;
                                
                                sum += input[inputIdx] * filter[(filterIdx * numChannels * filterSize * filterSize) + filterIdxOffset];
                            }
                        }
                    }
                }
    
                output[outIdx + (i * finalOutputWidth + j)] = sum;
            }
        }
    }
    private void updateFiltersWithMomentum(double[] filters, double[] dFilters, double[] vFilters, double learningRate) {
        // Calculate average gradient magnitude for adaptive scaling
        double avgGradMagnitude = 0.0;
        for (int i = 0; i < dFilters.length; i++) {
            avgGradMagnitude += Math.abs(dFilters[i]);
        }
        avgGradMagnitude /= dFilters.length;
        
        // Adjust learning rate based on gradient magnitude
        double adaptiveLR = learningRate;
        if (avgGradMagnitude > 0.1) {
            adaptiveLR = learningRate * (0.1 / avgGradMagnitude);
        }
        
        // Iterate over each filter
        for (int k = 0; k < numFilters; k++) {
            for (int c = 0; c < numChannels; c++) {
                int filterIndex = (k * numChannels + c) * filterSize * filterSize;
                
                for (int i = 0; i < filterSize * filterSize; i++) {
                    // Update vFilters with the momentum term
                    vFilters[filterIndex + i] = beta * vFilters[filterIndex + i] + (1 - beta) * dFilters[filterIndex + i];
                    
                    // Update the filter weights with adaptive learning rate
                    filters[filterIndex + i] -= adaptiveLR * vFilters[filterIndex + i];
                }
            }
        }
        
        if (super.getDebug() && adaptiveLR != learningRate) {
            System.out.println("Filter learning rate adjusted to: " + adaptiveLR + 
                              " (avg gradient: " + avgGradMagnitude + ")");
        }
    }
    
    private void updateBiasesWithMomentum(double[] biases, double[] dBiases, double[] vBiases, double learningRate) {
        // Calculate average gradient magnitude
        double avgGradMagnitude = 0.0;
        for (int i = 0; i < dBiases.length; i++) {
            avgGradMagnitude += Math.abs(dBiases[i]);
        }
        avgGradMagnitude /= dBiases.length;
        
        // Adjust learning rate based on gradient magnitude
        final double adaptiveLR = (avgGradMagnitude > 0.05) ? 
                                 learningRate * (0.05 / avgGradMagnitude) : 
                                 learningRate;
        
        IntStream.range(0, biases.length).parallel().forEach(i -> {
            vBiases[i] = beta * vBiases[i] + (1 - beta) * dBiases[i];
            biases[i] -= adaptiveLR * vBiases[i];
        });
        
        if (super.getDebug() && adaptiveLR != learningRate) {
            System.out.println("Bias learning rate adjusted to: " + adaptiveLR + 
                              " (avg gradient: " + avgGradMagnitude + ")");
        }
    }

    private void scaleFilterGradients() {
        // Calculate Frobenius norm of filter gradients
        double sumSquared = 0.0;
        double[] dFiltersMatrix = dFilters.getMatrix();
        
        for (int i = 0; i < dFiltersMatrix.length; i++) {
            sumSquared += dFiltersMatrix[i] * dFiltersMatrix[i];
        }
        
        double frobeniusNorm = Math.sqrt(sumSquared);
        
        // Scale gradients if norm exceeds threshold
        if (frobeniusNorm > filterGradientClipNorm && frobeniusNorm > 0) {
            // double scalingFactor = filterGradientClipNorm / frobeniusNorm;
            double scalingFactor = filterGradientClipNorm / (frobeniusNorm + 1e-10);
            
            for (int i = 0; i < dFiltersMatrix.length; i++) {
                dFiltersMatrix[i] *= scalingFactor;
            }
            
            if (super.getDebug()) {
                System.out.println("Filter gradients scaled by: " + scalingFactor + 
                                  " (norm: " + frobeniusNorm + ")");
            }
        }
    }
    
    private void scaleBiasGradients() {
        // Calculate L2 norm of bias gradients
        double sumSquared = 0.0;
        double[] dBiasesMatrix = dBiases.getMatrix();
        
        for (int i = 0; i < dBiasesMatrix.length; i++) {
            sumSquared += dBiasesMatrix[i] * dBiasesMatrix[i];
        }
        
        double l2Norm = Math.sqrt(sumSquared);
        
        // Scale gradients if norm exceeds threshold
        if (l2Norm > biasGradientClipNorm && l2Norm > 0) {
            double scalingFactor = biasGradientClipNorm / l2Norm;
            
            for (int i = 0; i < dBiasesMatrix.length; i++) {
                dBiasesMatrix[i] *= scalingFactor;
            }
            
            if (super.getDebug()) {
                System.out.println("Bias gradients scaled by: " + scalingFactor + 
                                  " (norm: " + l2Norm + ")");
            }
        }
    }
    
    private void scaleInputGradients() {
        // Calculate Frobenius norm of input gradients
        double sumSquared = 0.0;
        double[] dXmatrix = dX.getMatrix();
        
        for (int i = 0; i < dXmatrix.length; i++) {
            sumSquared += dXmatrix[i] * dXmatrix[i];
        }
        
        double frobeniusNorm = Math.sqrt(sumSquared);
        
        // Scale gradients if norm exceeds threshold
        if (frobeniusNorm > inputGradientClipNorm && frobeniusNorm > 0) {
            double scalingFactor = inputGradientClipNorm / frobeniusNorm;
            
            for (int i = 0; i < dXmatrix.length; i++) {
                dXmatrix[i] *= scalingFactor;
            }
            
            if (super.getDebug()) {
                System.out.println("Input gradients scaled by: " + scalingFactor + 
                                  " (norm: " + frobeniusNorm + ")");
            }
        }
    }

   @Override
    protected HashMap<String, JMatrix> getWeights() {
        HashMap<String, JMatrix> parameters = new HashMap<>();

        parameters.put("conv_2d_filters", filters);
        parameters.put("conv_2d_biases", biases);

        return parameters;
    }

    @Override
    protected void setBatchNorm(BatchNorm batchNorm) {
        batchNorm.build(numFilters);
        super.setBatchNorm(batchNorm);
    }


    @Override
    public JMatrix getOutput() {
       return Z;
    }


    @Override
    public JMatrix getGradient() {
        return dX;
    }


    @Override
    protected int[] getOutputShape() {
        int[] outputShape = null;
        if (Z != null) {
            outputShape =  Z.shape();
        }
        if (getActivation() != null) {
            getActivation().setOutputShape(outputShape);
        }
        if (getDropout() != null) {
            getDropout().setOutputShape(outputShape);
        }
        return outputShape;
    }

    @Override
    protected HashMap<String, Double> advancedStatistics() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'advancedStatistics'");
    }
}