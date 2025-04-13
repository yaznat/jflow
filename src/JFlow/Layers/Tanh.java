package JFlow.Layers;

import java.util.HashMap;
import java.util.stream.IntStream;

import JFlow.JMatrix;

class Tanh extends Activation{

    public Tanh() {
        super("tanh", 0);
    }
    @Override
    JMatrix applyActivation(JMatrix input) {
        double[] output = new double[input.size()];
        double[] inputMatrix = input.getMatrix();

        IntStream.range(0, input.size()).parallel().forEach(i -> {
            output[i] = Math.tanh(inputMatrix[i]);
        });
        return new JMatrix(output, input.length(), input.channels(), input.height(), input.width());
    }

    @Override
    JMatrix applyDActivation(JMatrix Z, JMatrix gradient) {
        double[] dZ = new double[gradient.size()];
        double[] gMatrix = gradient.getMatrix();
        double[] zMatrix = Z.getMatrix();

        IntStream.range(0, gradient.size()).parallel().forEach(i -> {
            double tanhVal = Math.tanh(zMatrix[i]);  
            double dTanh = 1 - tanhVal * tanhVal;  
            dZ[i] = gMatrix[i] * dTanh; 
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
