package JFlow.Layers;

import java.util.HashMap;
import java.util.stream.IntStream;

import JFlow.JMatrix;

class Dropout extends Component{
    private double alpha;
    private JMatrix dropoutMask;
    private double[] dropoutMaskFlat;
    private int[] outputShape;

    public Dropout(double alpha) {
        super("dropout", 0);
        this.alpha = alpha;
    }
    protected void setOutputShape(int[] outputShape) {
        this.outputShape = outputShape;
    }
    @Override
    protected int[] getOutputShape() {
        return outputShape;
    }
    protected double alpha(){
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

    // apply dropout in both forward and backward propagation
    public JMatrix applyDropout(JMatrix layer)  {
        // multiply with mask and scale nonzero results to keep sum ~ the same
        return dropoutMask.multiply(layer).multiply(1 + alpha);
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