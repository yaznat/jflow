package JFlow.Layers;

import java.util.stream.IntStream;

import JFlow.JMatrix;

class Softmax extends Activation{
    public Softmax(){}

    @Override
    public JMatrix applyActivation(JMatrix A) {
        // Assume flat
        int rows = A.length();
        int cols = A.channels() * A.height() * A.width();

        double[] aMatrix = A.getMatrix();
        double[] Z = new double[A.size()];

        // Compute softmax column-wise
        IntStream.range(0, cols).forEach(i -> {
            double max = Double.NEGATIVE_INFINITY;

            // Find max value in column
            for (int j = 0; j < rows; j++) {
                max = Math.max(aMatrix[j * cols + i], max);
            }

            double sum = 0;
            for (int j = 0; j < rows; j++) {
                sum += Math.exp(aMatrix[j * cols + i] - max);
            }

            for (int j = 0; j < rows; j++) {
                Z[j * cols + i] = Math.exp(aMatrix[j * cols + i] - max) / sum;
            }
        });
        return new JMatrix(Z, A.length(), A.channels(), A.height(), A.width());
    }

    @Override
    JMatrix applyDActivation(JMatrix Z, JMatrix gradient) {
        return Z.subtract(gradient);
    }

}
