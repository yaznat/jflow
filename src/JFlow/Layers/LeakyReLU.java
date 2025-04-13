package JFlow.Layers;

import java.util.HashMap;
import java.util.stream.IntStream;
import JFlow.JMatrix;

class LeakyReLU extends Activation{
    private double alpha;
    public LeakyReLU(double alpha) {
        super("leaky_re_lu", 0);
        this.alpha = alpha;
    }

    public double getAlpha() {
        return alpha;
    }

    @Override
    public JMatrix applyActivation(JMatrix input) {
        int length = input.size();
        double[] Z = new double[length];

        double[] A = input.getMatrix();

        IntStream.range(0, length).parallel().forEach(i -> {
            Z[i] = (A[i] > 0) ? A[i] : alpha * A[i];
        });

        return new JMatrix(Z, input.length(), input.channels(), input.height(), input.width());
    }

    @Override
    public JMatrix applyDActivation(JMatrix Z, JMatrix gradient) {
        if (gradient.size() != Z.size()) {
            throw new IllegalArgumentException(
                "Sizes " + gradient.size() + " and " + 
                Z.size() + 
                "cannot be broadcast together."
            );
        }

        int length = gradient.size();
        double[] dZ = new double[length];
        double[] gradientMatrix = gradient.getMatrix();
        double[] zMatrix = Z.getMatrix();

        IntStream.range(0, length).parallel().forEach(i -> {
            dZ[i] = (zMatrix[i] > 0) ? gradientMatrix[i] : gradientMatrix[i] * alpha;
        });

        return new JMatrix(dZ, Z.length(), Z.channels(), Z.height(), Z.width());
    }

    @Override
    protected HashMap<String, JMatrix> getWeights() {
        return new HashMap<>();
    }

    @Override
    protected HashMap<String, Double> advancedStatistics() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'advancedStatistics'");
    }
}
