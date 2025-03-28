package JFlow.Layers;

import java.util.stream.IntStream;

import JFlow.JMatrix;

class Tanh extends Activation{

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
    
}
