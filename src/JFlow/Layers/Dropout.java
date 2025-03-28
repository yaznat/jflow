package JFlow.Layers;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

import JFlow.JMatrix;
import JFlow.Utility;

class Dropout {
    private double alpha;
    private JMatrix dropoutMask;
    private double[] dropoutMaskFlat;
    public Dropout(double alpha) {
        this.alpha = alpha;
    }
    public double alpha(){
        return alpha;
    }
    // Set dropout mask. Return usually not needed.
    public JMatrix newDropoutMask(int inputSize, int outputSize) {
        double[] dropoutMaskMatrix = new double[inputSize * outputSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                dropoutMaskMatrix[i * outputSize + j] = (Math.random() < alpha) ? 0 : 1;
            }
        }
        dropoutMask = new JMatrix(dropoutMaskMatrix, inputSize, outputSize, 1, 1);
        return dropoutMask;
    }
    // Set dropout mask (flat). Return usually not needed.
    public double[] newDropoutMaskConv(int numFilters) {
        dropoutMaskFlat = new double[numFilters];

        IntStream.range(0, numFilters).parallel().forEach(i -> {
            dropoutMaskFlat[i] = (Math.random() < alpha) ? 0 : 1;
        });

        return dropoutMaskFlat;
    }

   // Apply dropout to a flat array representing the convolutional output
    public void applyDropoutConv(double[] layer, int numImages, int numFilters, int height, int width) {
        int featureMapSize = height * width;
        ForkJoinPool pool = ForkJoinPool.commonPool(); 

        List<Callable<Void>> tasks = new ArrayList<>();

        for (int img = 0; img < numImages; img++) {
            for (int filter = 0; filter < numFilters; filter++) {
                final int imageIdx = img;
                final int filterIdx = filter;
                tasks.add(() -> {
                    double mask = dropoutMaskFlat[filterIdx];
                    int startIdx = (imageIdx * numFilters + filterIdx) * featureMapSize;
                    for (int i = 0; i < featureMapSize; i++) {
                        layer[startIdx + i] *= mask / (1 - alpha);
                    }
                    return null;
                });
            }
        }

        try {
            pool.invokeAll(tasks);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // apply dropout in both forward and backward propagation
    public JMatrix applyDropout(JMatrix layer)  {
        // multiply with max and scale nonzero results
        return dropoutMask.multiply(layer);
    }
}