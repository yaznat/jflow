package JFlow.Layers;

import java.util.stream.IntStream;
import JFlow.JMatrix;

class MaxPool2D extends Layer {
    private int poolSize, stride;
    private int numImages, channels, imageHeight, imageWidth, outputHeight, outputWidth;
    private JMatrix output, gradient, lastInput;

    protected MaxPool2D(int poolSize, int stride) {
        super(0, "max_pool_2d");
        this.poolSize = poolSize;
        this.stride = stride;
    }

    @Override
    public void forward(JMatrix input, boolean training) {
        this.imageHeight = input.height();
        this.imageWidth = input.width();
        this.numImages = input.length();
        this.channels = input.channels();

        // Calculate output dimensions
        outputHeight = (imageHeight - poolSize) / stride + 1;
        outputWidth = (imageWidth - poolSize) / stride + 1;
 
        // Save time by avoiding memory reassignment if possible
        if (output == null || !input.isSameShapeAs(lastInput)) {
            output = new JMatrix(numImages, channels, outputHeight, outputWidth);
        } else {
            output.fill(0);
        }
        lastInput = input;

        double[] inputMatrix = input.getMatrix();
        double[] outputMatrix = output.getMatrix();

        // Perform max pooling
        IntStream.range(0, numImages).parallel().forEach(i -> {
            for (int c = 0; c < channels; c++) {
                int inputOffset = i * channels * imageHeight * imageWidth + c * imageHeight * imageWidth;
                int outputOffset = i * channels * outputHeight * outputWidth + c * outputHeight * outputWidth;

                maxPool2D(inputMatrix, inputOffset, outputMatrix, outputOffset);
            }
        });

        if (getNextLayer() != null) {
            getNextLayer().forward(new JMatrix(outputMatrix, numImages, channels, outputHeight, outputWidth), training);
        }
    }

    @Override
    public void backward(JMatrix dOutput, double learningRate) {
        // Save time by avoiding memory reassignment if possible
        if (gradient == null) {
            gradient = lastInput.copyDims(); // Use lastInput dimensions
        } else {
            gradient.fill(0);
        }

        double[] gradientMatrix = gradient.getMatrix();
        double[] dOutputMatrix = dOutput.getMatrix();
        double[] lastInputMatrix = lastInput.getMatrix();

        // Calculate maxpool gradients
        // IntStream.range(0, numImages).parallel().forEach(i -> {
        //     for (int c = 0; c < channels; c++) {
        //         int inputOffset = i * channels * imageHeight * imageWidth + c * imageHeight * imageWidth;
        //         int outputOffset = i * channels * outputHeight * outputWidth + c * outputHeight * outputWidth;

        //         backpropMaxPool2D(dOutputMatrix, inputOffset, gradientMatrix, outputOffset, lastInputMatrix); 
        //     }
        // });
        IntStream.range(0, numImages).parallel().forEach(i -> {
            for (int c = 0; c < channels; c++) {
                int inputOffset = i * channels * imageHeight * imageWidth + c * imageHeight * imageWidth;
                int outputOffset = i * channels * outputHeight * outputWidth + c * outputHeight * outputWidth;
        
                backpropMaxPool2D(dOutputMatrix, outputOffset, gradientMatrix, inputOffset, lastInputMatrix); 
            }
        });
        if (super.getDebug()) 
            System.out.println("MaxPool2D Max gradient " + gradient.max());

        if (getPreviousLayer() != null) {
            getPreviousLayer().backward(new JMatrix(gradientMatrix, numImages, channels, imageHeight, imageWidth), learningRate);
        }
    }

    // Perform max pooling on one image
    private void maxPool2D(double[] input, int inputOffset, double[] output, int outputOffset) {
        for (int sX = 0; sX < outputHeight; sX++) {
            for (int sY = 0; sY < outputWidth; sY++) {
                double max = Double.NEGATIVE_INFINITY;

                for (int poolX = 0; poolX < poolSize; poolX++) {
                    for (int poolY = 0; poolY < poolSize; poolY++) {
                        int x = sX * stride + poolX;
                        int y = sY * stride + poolY;
                        int idx = inputOffset + x * imageWidth + y;

                        if (input[idx] > max) {
                            max = input[idx];
                        }
                    }
                }
                output[outputOffset + sX * outputWidth + sY] = max;
            }
        }
    }

    // Calculate the max pool gradient for one image
    // private void backpropMaxPool2D(double[] dOutput, int inputOffset, double[] gradient, int outputOffset, double[] lastInput) {
    //     for (int sX = 0; sX < outputHeight; sX++) {
    //         for (int sY = 0; sY < outputWidth; sY++) {
    //             double max = Double.NEGATIVE_INFINITY;
    //             int maxIndex = 0;

    //             for (int poolX = 0; poolX < poolSize; poolX++) {
    //                 for (int poolY = 0; poolY < poolSize; poolY++) {
    //                     int x = sX * stride + poolX;
    //                     int y = sY * stride + poolY;
    //                     int idx = inputOffset + x * imageWidth + y;

    //                     if (lastInput[idx] > max) { 
    //                         max = lastInput[idx];
    //                         maxIndex = idx;
    //                     }
    //                 }
    //             }

    //             // Pass the gradient only to the max index
    //             gradient[maxIndex] += dOutput[outputOffset + sX * outputWidth + sY];
    //         }
    //     }
    // }
    private void backpropMaxPool2D(double[] dOutput, int dOutputOffset, double[] gradient, int gradientOffset, double[] lastInput) {
        for (int sX = 0; sX < outputHeight; sX++) {
            for (int sY = 0; sY < outputWidth; sY++) {
                double max = Double.NEGATIVE_INFINITY;
                int maxX = 0, maxY = 0;
    
                // Find max position
                for (int poolX = 0; poolX < poolSize; poolX++) {
                    for (int poolY = 0; poolY < poolSize; poolY++) {
                        int x = sX * stride + poolX;
                        int y = sY * stride + poolY;
                        int inputIdx = gradientOffset + x * imageWidth + y;
    
                        if (lastInput[inputIdx] > max) { 
                            max = lastInput[inputIdx];
                            maxX = x;
                            maxY = y;
                        }
                    }
                }
    
                // Pass the gradient only to the max position
                int maxIdx = gradientOffset + maxX * imageWidth + maxY;
                int dOutputIdx = dOutputOffset + sX * outputWidth + sY;
                gradient[maxIdx] += dOutput[dOutputIdx];
            }
        }
    }

    @Override
    public JMatrix getOutput() {
        return output;
    }

    @Override
    public JMatrix getGradient() {
        return null; 
    }
    
}
