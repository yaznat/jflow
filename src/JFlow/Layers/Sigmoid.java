package JFlow.Layers;

import java.util.stream.IntStream;

import JFlow.JMatrix;

class Sigmoid extends Activation{

    @Override
    JMatrix applyActivation(JMatrix input) {
        double[] output = new double[input.size()];
        double[] inputMatrix = input.getMatrix();

        IntStream.range(0, input.size()).parallel().forEach(i -> {
            output[i] = 1.0 / (1.0 + Math.exp(-inputMatrix[i]));
        });
        return new JMatrix(output, input.length(), input.channels(), input.height(), input.width());
    }

    @Override
    JMatrix applyDActivation(JMatrix Z, JMatrix gradient) {
        double[] dZ = new double[gradient.size()];
        double[] gMatrix = gradient.getMatrix();
        double[] zMatrix = Z.getMatrix();

        IntStream.range(0, gradient.size()).parallel().forEach(i -> {
            double sigmoidVal = 1.0 / (1.0 + Math.exp(-zMatrix[i]));
            double dSigmoid = sigmoidVal * (1 - sigmoidVal);
            dZ[i] = gMatrix[i] * dSigmoid;
        });
        return new JMatrix(dZ, Z.length(), Z.channels(), Z.height(), Z.width());
    }

    
}
