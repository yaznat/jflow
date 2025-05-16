package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapePreservingLayer;

public class Softmax extends ShapePreservingLayer{
    public Softmax() {
        super("softmax");
    }

    @Override
    public JMatrix forward(JMatrix A, boolean training) {
        // Assume flat
        int rows = A.length();
        int cols = A.channels() * A.height() * A.width();

        JMatrix Z = A.zerosLike();

        // Compute softmax column-wise
        IntStream.range(0, cols).parallel().forEach(i -> {
            float max = Float.NEGATIVE_INFINITY;

            // Find max value in column
            for (int j = 0; j < rows; j++) {
                max = Math.max(A.get(j * cols + i), max);
            }

            float sum = 0;
            for (int j = 0; j < rows; j++) {
                sum += (float)Math.exp(A.get(j * cols + i) - max);
            }

            for (int j = 0; j < rows; j++) {
                Z.set(j * cols + i, (float)Math.exp(A.get(j * cols + i) - max) / sum);
            }
        });
        return trackOutput(Z);
    }

    @Override
    public JMatrix backward(JMatrix gradient) {
        return trackGradient(getOutput().subtract(gradient));
    }
}
