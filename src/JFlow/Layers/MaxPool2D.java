package JFlow.Layers;

import java.util.stream.IntStream;

import JFlow.JMatrix;
import JFlow.Utility;

public class MaxPool2D extends Layer {
    private int poolSize, stride;
    private int numImages, channels, imageHeight, imageWidth, outputHeight, outputWidth;
    private double[] output, gradient, lastInput;

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

        double[] inputMatrix = input.getMatrix();
        lastInput = input.getMatrix();

        // Retrieve dimensions
        outputHeight = (imageHeight - poolSize) / stride + 1;
        outputWidth = (imageWidth - poolSize) / stride + 1;

        output = new double[numImages * channels * outputHeight * outputWidth];

        // Parallel max pooling
        IntStream.range(0, numImages).parallel().forEach(i -> {
            for (int c = 0; c < channels; c++) {
                int inputOffset = i * channels * imageHeight * imageWidth + c * imageHeight * imageWidth;
                int outputOffset = i * channels * outputHeight * outputWidth + c * outputHeight * outputWidth;

                maxPool2D(inputMatrix, inputOffset, output, outputOffset);
            }
        });

        if (getNextLayer() != null) {
            getNextLayer().forward(new JMatrix(output, numImages, channels, outputHeight, outputWidth), training);
        }
    }

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

    @Override
    public void backward(JMatrix dOutput, double learningRate) {
        gradient = new double[lastInput.length];
        double[] dOutputMatrix = dOutput.getMatrix();
        // System.out.println("Gradient length: " + gradient.length); 
        // System.out.println("dOutput length: " + dOutput.length); 
        // int expectedGradientSize = dOutput.length * poolSize * poolSize;
        // System.out.println("Expected gradient size: " + expectedGradientSize);
        // System.out.println("Actual gradient size: " + gradient.length);

        // Parallel backpropagation
        IntStream.range(0, numImages).parallel().forEach(i -> {
            for (int c = 0; c < channels; c++) {
                int inputOffset = i * channels * imageHeight * imageWidth + c * imageHeight * imageWidth;
                int outputOffset = i * channels * outputHeight * outputWidth + c * outputHeight * outputWidth;

                backpropMaxPool2D(dOutputMatrix, inputOffset, outputOffset); 
            }
        });
        if (super.getDebug()) 
            System.out.println("MaxPool2D Max gradient " + Utility.max(gradient));

        if (getPreviousLayer() != null) {
            // if (getPreviousLayer() instanceof Dense) {
            //     learningRate *= 0.85;
            // }
            getPreviousLayer().backward(new JMatrix(gradient, numImages, channels, imageHeight, imageWidth), learningRate);
        }
    }

    private void backpropMaxPool2D(double[] dOutput, int inputOffset, int outputOffset) {
        for (int sX = 0; sX < outputHeight; sX++) {
            for (int sY = 0; sY < outputWidth; sY++) {
                double max = Double.NEGATIVE_INFINITY;
                int maxIndex = 0;

                for (int poolX = 0; poolX < poolSize; poolX++) {
                    for (int poolY = 0; poolY < poolSize; poolY++) {
                        int x = sX * stride + poolX;
                        int y = sY * stride + poolY;
                        int idx = inputOffset + x * imageWidth + y;

                        if (lastInput[idx] > max) { 
                            max = lastInput[idx];
                            maxIndex = idx;
                        }
                    }
                }

                // Pass the gradient only to the max index
                gradient[maxIndex] += dOutput[outputOffset + sX * outputWidth + sY];
            }
        }
    }

    @Override
    public JMatrix getOutput() {
        return null;
    }

    @Override
    public JMatrix getGradient() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getGradient'");
    }
    
}
