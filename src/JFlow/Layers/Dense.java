package JFlow.Layers;

import JFlow.JMatrix;

class Dense extends Layer{
    private JMatrix weights, dWeights, vWeights, A, Z, dZ, lastInput, gOutput, biases, dBiases, vBiases;
    // For momentum updates
    private final double beta = 0.9;


    protected Dense(int inputSize, int outputSize) {
        super(inputSize * outputSize + outputSize, "dense");
        // Initialize weights and biases
        double[] weights = new double[outputSize * inputSize];
        double[] biases = new double[outputSize];

        // He Initialization
        double scale = Math.sqrt(2.0 / inputSize);

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[i * inputSize + j] = (Math.random() - 0.5) * scale;  
            }
            biases[i] = (Math.random() - 0.5) * 0.5;
        }

        this.weights = new JMatrix(weights, outputSize, inputSize, 1, 1);
        this.biases = new JMatrix(outputSize, 1, 1, 1);

        this.dWeights = new JMatrix(outputSize, inputSize, 1, 1);
        this.dBiases = new JMatrix(outputSize, 1, 1, 1);

        // Initialize vWeights and vBiases to 0
        vWeights = new JMatrix(outputSize, inputSize, 1, 1);
        vBiases = new JMatrix(outputSize, 1, 1, 1);
    }

    public double[] getWeights() {
        return weights.getMatrix();
    }
    public double[] getBiases() {
        return biases.getMatrix();
    }

    @Override
    public void forward(JMatrix input, boolean training) {
        if (getPreviousLayer() == null || !(getPreviousLayer() instanceof Dense)) {
            input = input.transpose2D();
        }
        // Store lastInput for backpropagation
        lastInput = input;

        // Calculate forward output
        try {
            A = weights.dot(input, true); // scaled dot product
        } catch (IllegalArgumentException e) {

            A = weights.transpose2D().dot(input, true);
        }
        applyBiasByRow(A, biases); 

        // Apply activation
        if (getActivation() != null) {
            Z = getActivation().applyActivation(A);
        } else {
            Z = A;
        }
        


        // Apply dropout
        if (getDropout() != null && training) {
            getDropout().newDropoutMask(Z.length(), Z.channels() * Z.height() * Z.width()); // Generate new mask
            Z = getDropout().applyDropout(Z);
        }

        if (getNextLayer() != null) {
            getNextLayer().forward(Z, training);
        }
    }
    @Override
    public void backward(JMatrix gradient, double learningRate) {

        if (getDebug()) {
            System.out.println("Dense");
            System.out.println("Input images:" + gradient.length());
            System.out.println("Input channels:" + gradient.channels());
            System.out.println("Input height:" + gradient.height());
            System.out.println("Input width:" + gradient.width());
        }

        // Calculate dActivation
        if (getActivation() != null) {
            dZ = getActivation().applyDActivation(Z, gradient);
        } else {
            dZ = gradient;
        }
        // Apply dropout
        if (super.getDropout() != null) {
            dZ = getDropout().applyDropout(dZ);
        }

        // Calculate dWeights and dBiases
        try {
            dWeights = dZ.dot(lastInput.transpose2D(), true);
            dBiases.setMatrix(dZ.sum0(true));
        } catch (IllegalArgumentException e) {
            dWeights = dZ.transpose2D().dot(lastInput.transpose2D(), true);
            dBiases.setMatrix(dZ.transpose2D().sum0(true));
        }

        // Normalize dWeights and dBiases
        int batchSize = lastInput.channels() * lastInput.height() * lastInput.width();
        dWeights = adaptiveGradientClip(weights, dWeights, 1e-2);
        dBiases = dBiases.multiply(1.0 / batchSize);
            
        if (getDebug()){
            System.out.println("Max dense weights: " + weights.max());
            System.out.println("Max dense biases: " + biases.max());
            System.out.println("Max dense dWeights: " + dWeights.max());
            System.out.println("Max dense dbiases: " + dBiases.max());
        }

        // Calculate velocity update
        vWeights = vWeights.multiply(beta).add(dWeights.multiply(1 - beta));
        vBiases = vBiases.multiply(beta).add(dBiases.multiply(1 - beta));


        // Apply gradients
        weights = weights.subtract(vWeights.multiply(learningRate));
        biases = biases.subtract(vBiases.multiply(learningRate));

        // Calculate loss w.r.t previous layer
        try {
            gOutput = weights.transpose2D().dot(dZ, true); // scaled
        } catch (IllegalArgumentException e) {
            gOutput = weights.transpose2D().dot(dZ.transpose2D(), true); // scaled
        }
         

        if (getPreviousLayer() != null) {
            getPreviousLayer().backward(gOutput, learningRate);
        }
    }

    private void applyBiasByRow(JMatrix A, JMatrix bias) {
        double[] aMatrix = A.getMatrix();
        double[] biasMatrix = bias.getMatrix();
        int rows = A.length();
        int cols = A.channels() * A.height() * A.width();
        for (int i = 0; i < rows; i++) { 
            for (int j = 0; j < cols; j++) { 
                aMatrix[i * cols + j] += biasMatrix[i]; 
            }
        }
    }
    
    // Adaptively clip with frobenius norm
    private JMatrix adaptiveGradientClip(JMatrix weights, JMatrix biases, double epsilon) {
        double weightNorm = weights.frobeniusNorm();
        double gradNorm = dWeights.frobeniusNorm();
        double maxNorm = Math.max(gradNorm, epsilon * weightNorm);

        if (gradNorm > maxNorm) {
            double scale = maxNorm / gradNorm;
            return dWeights.multiply(scale);
        }
        return dWeights;
    }


    @Override
    public JMatrix getOutput() {
        return Z;
    }

    @Override
    public JMatrix getGradient() {
        return gOutput;
    }
}



