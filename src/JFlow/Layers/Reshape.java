package JFlow.Layers;

import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

class Reshape extends Layer{
    private int newChannels, newHeight, newWidth, oldSize;

    public Reshape(int channels, int height, int width) {
        super(0, "reshape");
        this.newChannels = channels;
        this.newHeight = height;
        this.newWidth = width;

    }

    @Override
    public void forward(double[] input, boolean training, int numImages, int channels, int height, int width) {
        this.oldSize = channels * height * width;

        if (oldSize != newChannels * newHeight * newWidth) {
            throw new IllegalArgumentException("Reshape size mismatch: expected " +
                    (newChannels * newHeight * newWidth) + " but got " + oldSize);
        }

        if (getNextLayer() != null) {
            getNextLayer().forward(input, training, numImages, newChannels, newHeight, newWidth);
        }
    }

    @Override
    public void forward(double[][] input, boolean training) {
        int batchSize = input[0].length;  // Assuming input is transposed (channels * height * width, batchSize)
        int numChannels = 1;  // Assuming grayscale images
        int spatialSize = input.length;  // height * width
        int height = (int) Math.sqrt(spatialSize);
        int width = height;
    
        double[] flattened = new double[batchSize * numChannels * height * width];
    
        ForkJoinPool pool = new ForkJoinPool();
        pool.submit(() -> 
            IntStream.range(0, batchSize).parallel().forEach(i -> {
                for (int j = 0; j < spatialSize; j++) {
                    flattened[i * spatialSize + j] = input[j][i]; // Transpose back
                }
            })
        ).join();
        pool.shutdown();
        pool.close();
    
        forward(flattened, training, batchSize, numChannels, height, width);
    }
    

    @Override
    public void backward(double[] input, double learningRate, int batchSize, int channels, int height, int width) {
        int outputSize = batchSize * channels * height * width;
        
        if (super.getDebug()) {
            System.out.println("reshape");
            System.out.println("Input images:" + batchSize);
            System.out.println("Input channels:" + channels);
            System.out.println("Input height:" + height);
            System.out.println("Input width:" + width);
        }
        
        if (input.length != outputSize) {
            throw new IllegalArgumentException("Input gradient size mismatch in Reshape layer.");
        }

        // Restore original shape (before forward reshape)
        double[][] output = new double[batchSize][oldSize]; // changed from chanels * height * width (incorrect) to oldsize

        // Parallel processing for performance
        IntStream.range(0, batchSize).parallel().forEach(i -> 
            System.arraycopy(input, i * oldSize, output[i], 0, oldSize)
        );

        // Pass to the previous layer
        if (getPreviousLayer() != null) {
            getPreviousLayer().backward(output, learningRate);
        }
    }

    @Override
    public double[][] getOutput() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getOutput'");
    }

    @Override
    public void backward(double[][] input, double learningRate) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'backward'");
    }

    @Override
    public double[][] getGradient() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getGradient'");
    }
    
}
