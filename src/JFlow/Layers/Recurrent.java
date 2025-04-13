package JFlow.Layers;

import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import JFlow.JMatrix;
class Recurrent extends Layer{
    private JMatrix Wx, Wh, biases, A, Z, hPrev, inputGradient;
    private int inputSize, hiddenSize;

    protected Recurrent(int inputSize, int hiddenSize) {
        super("recurrent", inputSize * hiddenSize + hiddenSize * hiddenSize + hiddenSize);
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        Random random = new Random();
        // Xavier initialization
        double[] Wx = new double[hiddenSize * inputSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                Wx[i * inputSize + j] = random.nextGaussian(0, 
                    Math.sqrt((1.0 / ((hiddenSize + inputSize) / 2))));
            }
        }
        double[] Wh = new double[hiddenSize * hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                Wh[i * hiddenSize + j] = random.nextGaussian(0, 
                    Math.sqrt((1.0 / hiddenSize)));
            }
        }
        double[] biases = new double[hiddenSize];

        this.Wx = new JMatrix(Wx, hiddenSize, inputSize, 1, 1);
        this.Wh = new JMatrix(Wh, hiddenSize, hiddenSize, 1, 1);
        this.biases = new JMatrix(biases, hiddenSize, 1, 1, 1);
    }

    @Override
    protected int channels() {
        return -1;
    }

    @Override
    protected void forward(JMatrix input, boolean training) {
        int batchSize = input.length();
        int sequenceLength = input.height();
        // Cache values for backpropagation
        A = new JMatrix(batchSize, 1, sequenceLength, hiddenSize);
        Z = new JMatrix(batchSize, 1, sequenceLength, hiddenSize);

        IntStream.range(0, batchSize).parallel().forEach(i -> {
            hPrev = new JMatrix(hiddenSize, 1, 1, 1);
            // Loop forward in time
            for (int t = 0; t < sequenceLength; t++) {
                JMatrix x_t = input.getWrapped(i, t);

                JMatrix Wx_x = Wx.dot(x_t, true); 
                JMatrix Wh_h = Wh.dot(hPrev, true);

                JMatrix a_t = Wx_x.add(Wh_h).add(biases);

                JMatrix z_t = null;
                if (getActivation() != null) {
                    z_t = getActivation().applyActivation(a_t);
                } else {
                    z_t = a_t;
                }

                int offset = i * (sequenceLength * hiddenSize) + t * hiddenSize;

                System.arraycopy(a_t.getMatrix(), 0, A.getMatrix(), offset, hiddenSize);
                System.arraycopy(z_t.getMatrix(), 0, Z.getMatrix(), offset, hiddenSize);

                hPrev = z_t;
            }
        });
    }

    @Override
    protected void backward(JMatrix input, double learningRate) {
        int batchSize = input.length();
        int sequenceLength = input.height();

        // Initialize gradients
        JMatrix dWxGlobal = Wx.zerosLike();
        JMatrix dWhGlobal = Wh.zerosLike();
        JMatrix dbiasesGlobal = biases.zerosLike();
        JMatrix dInput = input.zerosLike();

        // Use parallel stream with thread-local accumulators
        List<GradientTriple> threadResults = IntStream.range(0, batchSize).parallel()
            .mapToObj(i -> {
                JMatrix dWxLocal = Wx.zerosLike();
                JMatrix dWhLocal = Wh.zerosLike();
                JMatrix dbiasesLocal = biases.zerosLike();
                JMatrix dInputLocal = input.zerosLike();
                JMatrix dH_next = new JMatrix(hiddenSize, 1, 1, 1); // zero init

                for (int t = sequenceLength - 1; t >= 0; t--) {
                    JMatrix upstream = input.getWrapped(i, t);
                    JMatrix z_t = Z.getWrapped(i, t);
                    JMatrix a_t = A.getWrapped(i, t);

                    JMatrix dA_t = (getActivation() != null)
                        ? getActivation().applyDActivation(z_t, upstream.add(dH_next))
                        : upstream.add(dH_next);

                    JMatrix x_t = input.getWrapped(i, t);
                    JMatrix h_prev_t = (t > 0) ? Z.getWrapped(i, t - 1) : new JMatrix(hiddenSize, 1, 1, 1);

                    dWxLocal = dWxLocal.add(dA_t.dot(x_t.transpose2D(), true));
                    dWhLocal = dWhLocal.add(dA_t.dot(h_prev_t.transpose2D(), true));
                    dbiasesLocal = dbiasesLocal.add(dA_t);

                    dH_next = Wh.transpose2D().dot(dA_t, true);
                    JMatrix dx_t = Wx.transpose2D().dot(dA_t, true);
                    dInputLocal.set(i, t, dx_t.getMatrix());
                }

                return new GradientTriple(dWxLocal, dWhLocal, dbiasesLocal, dInputLocal);
            })
            .toList();

        // Merge thread-local gradients into global ones
        for (GradientTriple g : threadResults) {
            dWxGlobal = dWxGlobal.add(g.dWx);
            dWhGlobal = dWhGlobal.add(g.dWh);
            dbiasesGlobal = dbiasesGlobal.add(g.dbiases);
            dInput = dInput.add(g.dInput); // or do this per batch index if more control is needed
        }

        this.inputGradient = dInput;

        // Optionally update weights here using learningRate
        Wx = Wx.subtract(dWxGlobal.multiply(learningRate));
        Wh = Wh.subtract(dWhGlobal.multiply(learningRate));
        biases = biases.subtract(dbiasesGlobal.multiply(learningRate));
    }
    

    @Override
    protected JMatrix getOutput() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getOutput'");
    }

    @Override
    protected JMatrix getGradient() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getGradient'");
    }

    @Override
    protected HashMap<String, JMatrix> getWeights() {
        HashMap<String, JMatrix> parameters = new HashMap<>();

        parameters.put("recurrent_weightsX", Wx);
        parameters.put("recurrent_weightsH", Wh);
        parameters.put("recurrent_biases", biases);

        return parameters;
    }

    @Override
    protected int[] getOutputShape() {
        return null;
    }

    private static class GradientTriple {
        JMatrix dWx, dWh, dbiases, dInput;
        GradientTriple(JMatrix dWx, JMatrix dWh, JMatrix dbiases, JMatrix dInput) {
            this.dWx = dWx;
            this.dWh = dWh;
            this.dbiases = dbiases;
            this.dInput = dInput;
        }
    }

    @Override
    protected HashMap<String, Double> advancedStatistics() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'advancedStatistics'");
    }
    
}

