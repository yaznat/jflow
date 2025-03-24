package JFlow.Layers;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

import JFlow.JMatrix;

public class Upsampling2D extends Layer{
    private int scaleFactor;
    private double[] output = new double[0], gradient = new double[0];

    public Upsampling2D(int scaleFactor) {
        super(0, "up_sampling_2d");
        this.scaleFactor = scaleFactor;
    }


    @Override
    // Expand input by scalefactor
    public void forward(JMatrix input, boolean training) {
    int numImages = input.length();
    int channels = input.channels();
    int height = input.height();
    int width = input.width();

    double[] inputMatrix = input.getMatrix();
    
    int newHeight = height * scaleFactor;
    int newWidth = width * scaleFactor;
    if (output.length != input.size()) {
        output = new double[numImages * channels * newHeight * newWidth];
    }

    IntStream.range(0, numImages * channels).parallel().forEach(index -> {
        int i = index / channels;  
        int c = index % channels;  

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                // Map input value to square region of output values
                double value = inputMatrix[i * channels * height * width + c * height * width + h * width + w];

                for (int a = 0; a < scaleFactor; a++) {
                    for (int b = 0; b < scaleFactor; b++) {
                        int newH = h * scaleFactor + a;
                        int newW = w * scaleFactor + b;
                        output[i * channels * newHeight * newWidth + c * newHeight * newWidth + newH * newWidth + newW] = value;
                    }
                }
            }
        }
    });

    if (getNextLayer() != null) {
        getNextLayer().forward(new JMatrix(output, numImages, channels, newHeight, newWidth), training);
    }
}

    @Override
    // Shrink input gradient by scalefactor, summing regions
    public void backward(JMatrix input, double learningRate) {
        int numImages = input.length();
        int channels = input.channels();
        int height = input.height();
        int width = input.width();
        double[] inputMatrix = input.getMatrix(); 
        if (super.getDebug()) {
            System.out.println("upsampling2d");
            System.out.println("Input images:" + numImages);
            System.out.println("Input channels:" + channels);
            System.out.println("Input height:" + height);
            System.out.println("Input width:" + width);
        }
        int newHeight = height / scaleFactor;
        int newWidth = width / scaleFactor;
        if (gradient.length != input.size()) {
            gradient = new double[numImages * channels * newHeight * newWidth];
        }
    
        IntStream.range(0, numImages * channels).parallel().forEach(index -> {
            int i = index / channels;  
            int c = index % channels;  
    
            for (int newH = 0; newH < newHeight; newH++) {
                for (int newW = 0; newW < newWidth; newW++) {
                    int outputIndex = i * channels * newHeight * newWidth + c * newHeight * newWidth + newH * newWidth + newW;
    
                    double sum = 0.0; 
                    for (int a = 0; a < scaleFactor; a++) {
                        for (int b = 0; b < scaleFactor; b++) {
                            int h = newH * scaleFactor + a;
                            int w = newW * scaleFactor + b;
                            int inputIndex = i * channels * height * width + c * height * width + h * width + w;
                            sum += inputMatrix[inputIndex];  // Sum gradients from upscaled region
                        }
                    }
                    gradient[outputIndex] = sum;
                }
            }
        });
    
        if (getPreviousLayer() != null) {
            getPreviousLayer().backward(new JMatrix(gradient, numImages, channels, newHeight, newWidth), learningRate);
        }
    }

    @Override
    public JMatrix getOutput() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getOutput'");
    }


    @Override
    public JMatrix getGradient() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getGradient'");
    }
    
}
