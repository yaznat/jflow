package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

import java.util.concurrent.ThreadLocalRandom;

public class Conv2D extends TrainableLayer {
    private JMatrix filters;
    private JMatrix dFilters;
    private JMatrix lastInput;
    private JMatrix biases;
    private JMatrix dBiases;

    private int numFilters;
    private int filterSize;
    private int stride;
    private int numChannels;
    private int inputHeight;
    private int inputWidth;
    private int numImages;

    private String padding;
    // Hyperparameters for gradient clipping
    final double epsilon = 1e-8;         // Small constant for numerical stability
    final double clipThreshold = 5.0;   // Global gradient clipping threshold

    public Conv2D(int numFilters, int filterSize, int stride, String padding) {
        super("conv_2d");
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.stride = stride;
        this.padding = padding;
        if (!(padding.equals("same_padding") || padding.equals("valid_padding"))) {
            throw new IllegalArgumentException("Only same_padding and valid_padding allowed.");
        }
    }
    public Conv2D(int numFilters, int filterSize, int stride, String padding, int[] inputShape) {
        super("conv_2d");
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.stride = stride;
        
        
        if (!(padding.equals("same_padding") || padding.equals("valid_padding"))) {
            throw new IllegalArgumentException("Only same_padding and valid_padding allowed.");
        }
        this.padding = padding;
        if (inputShape.length != 3) {
            throw new IllegalArgumentException(
                "Conv2D input shape should have 3 dimensions. Got: "
                + inputShape.length + "."
            );
        }
        setInputShape(inputShape);
    }

    @Override
    public void build(int IDnum) {
        super.build(IDnum);
        if (internalGetInputShape() != null) {
            this.numChannels = internalGetInputShape()[0];
        } else {
            if (getPreviousLayer() == null) {
                throw new IllegalStateException(
                    "In " + this.getClass().getSimpleName() + 
                    ": Cannot build the first layer without an input shape."
                );
            }
            this.numChannels = getPreviousLayer().outputShape()[1];
        }
        setNumTrainableParameters(numFilters * numChannels * filterSize * filterSize + numFilters);
        // He initialization

        double stdDev = Math.sqrt(2.0 / (numChannels * filterSize * filterSize));

        double filterScale = 1.0;

        int filterSizeTotal = numFilters * numChannels * filterSize * filterSize;
        float[] filters = new float[filterSizeTotal];

        IntStream.range(0, filterSizeTotal).parallel().forEach(i -> {
            filters[i] = (float)(ThreadLocalRandom.current().nextGaussian() * stdDev * filterScale);
        });

        this.filters = new JMatrix(filters, numFilters, numChannels, filterSize, filterSize, "filters");

        // Initialize biases to 0
        biases = new JMatrix(numFilters, 1, 1, 1, "biases");

        // Set name for debug capability
        dFilters = new JMatrix(numFilters, numChannels, filterSize, filterSize, "dFilters");
        dBiases = new JMatrix(numFilters, 1, 1, 1, "dBiases");
        
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        lastInput = input;
        this.inputHeight = input.height();
        this.inputWidth = input.width();
        this.numImages = input.length();
    
        // Calculate output dimensions based on padding and stride
        int outputHeight, outputWidth;
        if (padding.equals("same_padding")) {
            outputHeight = (int)Math.ceil((double)inputHeight / stride);
            outputWidth = (int)Math.ceil((double)inputWidth / stride);
        } else { // valid padding
            outputHeight = (inputHeight - filterSize) / stride + 1;
            outputWidth = (inputWidth - filterSize) / stride + 1;
        }
    
        // Initialize output matrix with proper dimensions
        JMatrix A = new JMatrix(numImages, numFilters, outputHeight, outputWidth);
        
        // Calculate forward output
        if (numImages <= Runtime.getRuntime().availableProcessors() / 2) {
            // For each image in the batch
            for (int imageIndex = 0; imageIndex < numImages; imageIndex++) {
                final int imgIdx = imageIndex;
                int startIdx = imgIdx * numChannels * inputHeight * inputWidth;
                
                // Parallelize across filters
                IntStream.range(0, numFilters).parallel().forEach(filterIndex -> {
                    int outputIdx = (imgIdx * numFilters + filterIndex) * outputHeight * outputWidth;
                    convolveWithKernel(A.getMatrix(), outputIdx, input.getMatrix(), startIdx,
                            filters.getMatrix(), filterIndex, biases.get(filterIndex), padding);
                });
            }
        } else {
            // Parallelize across batch for larger batch sizes
            IntStream.range(0, numImages).parallel().forEach(imageIndex -> {
                for (int filterIndex = 0; filterIndex < numFilters; filterIndex++) {
                    int startIdx = imageIndex * numChannels * inputHeight * inputWidth;
                    int outputIdx = (imageIndex * numFilters + filterIndex) * outputHeight * outputWidth;
                    convolveWithKernel(A.getMatrix(), outputIdx, input.getMatrix(), startIdx,
                            filters.getMatrix(), filterIndex, biases.get(filterIndex), padding);
                }
            });
        }
       
        return trackOutput(A);
    }

    @Override
    public JMatrix backward(JMatrix input) {
        // Calculate output dimensions based on padding and stride
        int outputHeight, outputWidth;
        if (padding.equals("same_padding")) {
            outputHeight = (int)Math.ceil((double)inputHeight / stride);
            outputWidth = (int)Math.ceil((double)inputWidth / stride);
        } else { // valid padding
            outputHeight = (inputHeight - filterSize) / stride + 1;
            outputWidth = (inputWidth - filterSize) / stride + 1;
        }
        
        // Initialize dX with proper dimensions
        JMatrix dX = new JMatrix(numImages, numChannels, inputHeight, inputWidth);
        
        // Calculate gradients in batch
    
        // For each filter
        IntStream.range(0, numFilters).parallel().forEach(k -> {
            // Calculate bias gradients
            float biasGrad = 0;
            for (int i = 0; i < numImages; i++) {
                int dZFilterOffset = (i * numFilters + k) * outputHeight * outputWidth;
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        biasGrad += input.get(dZFilterOffset + oh * outputWidth + ow);
                    }
                }
            }
            dBiases.set(k, biasGrad);
            
            // Calculate filter gradients
            for (int c = 0; c < numChannels; c++) {
                int filterChannelOffset = ((k * numChannels) + c) * filterSize * filterSize;
                
                for (int fh = 0; fh < filterSize; fh++) {
                    for (int fw = 0; fw < filterSize; fw++) {
                        float filterGrad = 0;
                        
                        // Accumulate gradients from all images in batch
                        for (int i = 0; i < numImages; i++) {
                            int inputChannelOffset = (i * numChannels + c) * inputHeight * inputWidth;
                            int dZFilterOffset = (i * numFilters + k) * outputHeight * outputWidth;
                            
                            for (int oh = 0; oh < outputHeight; oh++) {
                                for (int ow = 0; ow < outputWidth; ow++) {
                                    // Map output coordinates to input coordinates
                                    int ih_base, iw_base;
                                    
                                    if (padding.equals("same_padding")) {
                                        // Calculate padding size
                                        int padTotal_h = Math.max(0, (outputHeight - 1) * stride + filterSize - inputHeight);
                                        int padTotal_w = Math.max(0, (outputWidth - 1) * stride + filterSize - inputWidth);
                                        int padTop = padTotal_h / 2;
                                        int padLeft = padTotal_w / 2;
                                        
                                        // Adjust for stride and padding
                                        ih_base = oh * stride - padTop;
                                        iw_base = ow * stride - padLeft;
                                    } else { // valid padding
                                        ih_base = oh * stride;
                                        iw_base = ow * stride;
                                    }
                                    
                                    // The position in input corresponding to this filter position
                                    int ih = ih_base + fh;
                                    int iw = iw_base + fw;
                                    
                                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                        int inputIdx = inputChannelOffset + (ih * inputWidth + iw);
                                        int dZIdx = dZFilterOffset + (oh * outputWidth + ow);
                                        filterGrad += lastInput.get(inputIdx) * input.get(dZIdx);
                                    }
                                }
                            }
                        }
                        
                        dFilters.set(filterChannelOffset + (fh * filterSize + fw), filterGrad);
                    }
                }
            }
        });
        
        // Calculate input gradients (dX)
        // Parallelize across input channels and spatial blocks
        final int TILE_SIZE = 32;
        int numTilesH = (inputHeight + TILE_SIZE - 1) / TILE_SIZE;
        int numTilesW = (inputWidth + TILE_SIZE - 1) / TILE_SIZE;
    
        // Create parallel tasks for each channel+tile combination
        IntStream.range(0, numChannels * numTilesH * numTilesW).parallel().forEach(taskIdx -> {
            int c = taskIdx / (numTilesH * numTilesW);
            int tileIdx = taskIdx % (numTilesH * numTilesW);
            int tileH = tileIdx / numTilesW;
            int tileW = tileIdx % numTilesW;
            
            // Calculate tile boundaries
            int ih_start = tileH * TILE_SIZE;
            int ih_end = Math.min(ih_start + TILE_SIZE, inputHeight);
            int iw_start = tileW * TILE_SIZE;
            int iw_end = Math.min(iw_start + TILE_SIZE, inputWidth);
            
            // Calculate padding for position mapping
            int padTop = 0, padLeft = 0;
            if (padding.equals("same_padding")) {
                int padTotal_h = Math.max(0, (outputHeight - 1) * stride + filterSize - inputHeight);
                int padTotal_w = Math.max(0, (outputWidth - 1) * stride + filterSize - inputWidth);
                padTop = padTotal_h / 2;
                padLeft = padTotal_w / 2;
            }
            
            // Process for all images in the batch
            for (int i = 0; i < numImages; i++) {
                int dXChannelOffset = (i * numChannels + c) * inputHeight * inputWidth;
                
                // Compute gradients for this tile
                for (int ih = ih_start; ih < ih_end; ih++) {
                    for (int iw = iw_start; iw < iw_end; iw++) {
                        float sum = 0;
                        
                        // For each filter
                        for (int k = 0; k < numFilters; k++) {
                            int filterChannelBaseOffset = (k * numChannels + c) * filterSize * filterSize;
                            int dZFilterOffset = (i * numFilters + k) * outputHeight * outputWidth;
                            
                            // For each filter position that affects this input pixel
                            for (int fh = 0; fh < filterSize; fh++) {
                                for (int fw = 0; fw < filterSize; fw++) {
                                    // Rotate filter by 180 degrees for the transposed convolution
                                    int rotatedFh = filterSize - 1 - fh;
                                    int rotatedFw = filterSize - 1 - fw;
                                    int filterPos = filterChannelBaseOffset + (rotatedFh * filterSize + rotatedFw);
                                    
                                    // Calculate corresponding output position with padding and stride
                                    int oh, ow;
                                    
                                    if (padding.equals("same_padding")) {
                                        // Determine which output cell affects this input position
                                        oh = (ih - rotatedFh + padTop) / stride;
                                        ow = (iw - rotatedFw + padLeft) / stride;
                                        
                                        // Check for stride boundary
                                        if ((ih - rotatedFh + padTop) % stride != 0 || 
                                            (iw - rotatedFw + padLeft) % stride != 0) {
                                            continue;
                                        }
                                    } else { // valid padding
                                        // For valid padding
                                        oh = (ih - rotatedFh) / stride;
                                        ow = (iw - rotatedFw) / stride;
                                        
                                        // Check for stride boundary
                                        if ((ih - rotatedFh) % stride != 0 || 
                                            (iw - rotatedFw) % stride != 0) {
                                            continue;
                                        }
                                    }
                                    
                                    if (oh >= 0 && oh < outputHeight && ow >= 0 && ow < outputWidth) {
                                        int dZPos = dZFilterOffset + (oh * outputWidth + ow);
                                        sum += filters.get(filterPos) * input.get(dZPos);
                                    }
                                }
                            }
                        }
                        
                        int dXPos = dXChannelOffset + (ih * inputWidth + iw);
                        dX.set(dXPos, sum);
                    }
                }
            }
        });
        
        // Apply adaptive gradient scaling
        adaptiveScale(dFilters, dBiases, dX);
    
        return trackGradient(dX);
    }
    
    // Apply convolution to one image at a time
    private void convolveWithKernel(float[] output, int outIdx, float[] input, int inIdx, 
                               float[] kernel, int filterIdx, float bias, String padding) {
        // Calculate padding and output dimensions
        int outputHeight, outputWidth;
        int padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
        
        if (padding.equals("same_padding")) {
            int padTotal_h = Math.max(0, (inputHeight - 1) * stride + filterSize - inputHeight);
            int padTotal_w = Math.max(0, (inputWidth - 1) * stride + filterSize - inputWidth);
            
            padTop = padTotal_h / 2;
            padBottom = padTotal_h - padTop;
            padLeft = padTotal_w / 2;
            padRight = padTotal_w - padLeft;
            
            outputHeight = (int)Math.ceil((double)inputHeight / stride);
            outputWidth = (int)Math.ceil((double)inputWidth / stride);
        } else { // valid padding
            outputHeight = (inputHeight - filterSize) / stride + 1;
            outputWidth = (inputWidth - filterSize) / stride + 1;
        }
        
        // Process each position in the output feature map
        for (int oh = 0; oh < outputHeight; oh++) {
            for (int ow = 0; ow < outputWidth; ow++) {
                float sum = bias;
                
                // For each input channel
                for (int c = 0; c < numChannels; c++) {
                    int inputChannelOffset = inIdx + (c * inputHeight * inputWidth);
                    int filterChannelOffset = (filterIdx * numChannels + c) * filterSize * filterSize;
                    
                    // For each element in the filter
                    for (int fh = 0; fh < filterSize; fh++) {
                        for (int fw = 0; fw < filterSize; fw++) {
                            // Calculate input position
                            int ih, iw;
                            
                            if (padding.equals("same_padding")) {
                                ih = oh * stride + fh - padTop;
                                iw = ow * stride + fw - padLeft;
                            } else { // valid padding
                                ih = oh * stride + fh;
                                iw = ow * stride + fw;
                            }
                            
                            // Check bounds and accumulate weighted value if within bounds
                            if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                                int inputPos = inputChannelOffset + (ih * inputWidth + iw);
                                int filterPos = filterChannelOffset + (fh * filterSize + fw);
                                
                                sum += input[inputPos] * kernel[filterPos];
                            }
                        }
                    }
                }
                
                // Store result in output
                output[outIdx + (oh * outputWidth + ow)] = sum;
            }
        }
    }

    @Override
    public void updateParameters(JMatrix[] parameterUpdates) {
        filters.subtractInPlace(parameterUpdates[0]);
        biases.subtractInPlace(parameterUpdates[1]);
    }

    private void adaptiveScale(JMatrix dFilters, JMatrix dBiases, JMatrix dX) {
        // Clip filter gradients
        double filterNorm = filters.frobeniusNorm();
        double dFilterNorm = dFilters.frobeniusNorm();
        double maxFilterNorm = epsilon * filterNorm;
        
        if (dFilterNorm > maxFilterNorm) {
            double scaleFilter = maxFilterNorm / dFilterNorm;
            dFilters.multiplyInPlace(scaleFilter);
        }
        
        // Clip bias gradients
        double biasNorm = biases.frobeniusNorm();
        double dBiasNorm = dBiases.frobeniusNorm();
        double maxBiasNorm = Math.max(dBiasNorm, epsilon * biasNorm);
        
        if (dBiasNorm > maxBiasNorm) {
            double scaleBias = maxBiasNorm / dBiasNorm;
            dBiases.multiplyInPlace(scaleBias);
        }
        
        // Apply gradient scaling for input gradients if needed
        double dXNorm = dX.frobeniusNorm();
        if (dXNorm > clipThreshold) {
            double scale = clipThreshold / (dXNorm + epsilon);
            dX.multiplyInPlace(scale);
        }
    }
    
    @Override
    public JMatrix[] getWeights() {
        return new JMatrix[]{filters, biases};
    }

    @Override
    public JMatrix[] getParameterGradients() {
        return new JMatrix[]{dFilters, dBiases};
    }


    @Override
    public int[] outputShape() {
        int[] outputShape = null;
        if (getOutput() != null) {
            outputShape = getOutput().shape();
        } else {
            int[] prevShape;
            if (getPreviousLayer() == null) {
                int[] inputShape = internalGetInputShape();
                prevShape = new int[]{-1, inputShape[0], inputShape[1], inputShape[2]};
            } else {
                prevShape = getPreviousLayer().outputShape().clone();
            }
            if (padding.equals("same_padding")) {
                outputShape = new int[]{
                    prevShape[0],
                    numFilters,
                    (int)Math.ceil((double)prevShape[2] / stride),
                    (int)Math.ceil((double)prevShape[3] / stride)
                };
            } else { // valid padding
                outputShape = new int[]{
                    prevShape[0],
                    numFilters,
                    (prevShape[2] - filterSize) / stride + 1,
                    (prevShape[3] - filterSize) / stride + 1
                };
            }
        }
        return outputShape;
    }
   
}