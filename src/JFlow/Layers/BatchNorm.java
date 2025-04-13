package JFlow.Layers;

import java.util.HashMap;
import java.util.stream.IntStream;
import JFlow.JMatrix;

public class BatchNorm extends Component {
    private int featureSize;
    private double epsilon = 1e-5;
    private double momentum = 0.9;
    private JMatrix gamma;
    private JMatrix beta;
    private JMatrix runningMean;
    private JMatrix runningVar;
    private JMatrix batchMean;
    private JMatrix batchVar;
    private JMatrix xHat;
    private JMatrix input;
    
    // Pre-allocated matrices for backward pass
    private JMatrix dGamma;
    private JMatrix dBeta;
    private JMatrix dx;
    private JMatrix dxHat;
    private JMatrix dxHatSum;
    private JMatrix dxHatXhatSum;

    public BatchNorm() {
        super("batch_norm", 0);
    }
    public void build(int featureSize) {
        this.featureSize = featureSize;

        setNumTrainableParameters(featureSize * 2);

        // Initialize gamma values as 1.0
        this.gamma = new JMatrix(1, featureSize, 1, 1);
        gamma.fill(1.0);
        
        // Initialize beta values as 0
        this.beta = new JMatrix(1, featureSize, 1, 1);
        
        this.runningMean = new JMatrix(1, featureSize, 1, 1);
        this.runningVar = new JMatrix(1, featureSize, 1, 1);
        runningVar.fill(1.0);
        
        this.dGamma = new JMatrix(1, featureSize, 1, 1);
        this.dBeta = new JMatrix(1, featureSize, 1, 1);
        this.dxHatSum = new JMatrix(1, featureSize, 1, 1);
        this.dxHatXhatSum = new JMatrix(1, featureSize, 1, 1);

    }

    public JMatrix forward(JMatrix input, boolean training) {
        this.input = input;
        
        // Ensure dx and dxHat have the right dimensions
        if (dx == null || dx.length() != input.length() || dx.channels() != input.channels()) {
            dx = new JMatrix(input.length(), input.channels(), input.height(), input.width());
            dxHat = new JMatrix(input.length(), input.channels(), input.height(), input.width());
        }
        
        if (input.channels() != featureSize) {
            System.out.println("Warning: BatchNorm feature size doesn't match input channels");
        }
        
        if (training) {
            // Compute batch statistics
            batchMean = calcMean(input);
            batchVar = calcVariance(input, batchMean);
            
            // Update running averages
            runningMean = runningMean.multiply(momentum).add(batchMean.multiply(1 - momentum));
            runningVar = runningVar.multiply(momentum).add(batchVar.multiply(1 - momentum));
            
            // Normalize
            xHat = normalize(input, batchMean, batchVar);
        } else {
            // Normalize using running averages
            xHat = normalize(input, runningMean, runningVar);
        }
        
        // Scale and shift
        JMatrix output = scaleAndShift(xHat, gamma, beta);
        
        return output;
    }
    
    private JMatrix calcMean(JMatrix input) {
        JMatrix mean = new JMatrix(1, featureSize, 1, 1);
        double[] meanData = mean.getMatrix();
        double[] inputData = input.getMatrix();
        
        int batchSize = input.length();
        int channels = input.channels();
        int height = input.height();
        int width = input.width();
        int spatialSize = height * width;
        
        // Parallelize across channels
        IntStream.range(0, channels)
            .parallel()
            .forEach(c -> {
                double sum = 0;
                for (int n = 0; n < batchSize; n++) {
                    int batchOffset = n * channels * height * width;
                    int channelOffset = c * height * width;
                    
                    for (int h = 0; h < height; h++) {
                        int rowOffset = h * width;
                        
                        for (int w = 0; w < width; w++) {
                            int idx = batchOffset + channelOffset + rowOffset + w;
                            sum += inputData[idx];
                        }
                    }
                }
                meanData[c] = sum / (batchSize * spatialSize);
            });
        
        return mean;
    }
    
    private JMatrix calcVariance(JMatrix input, JMatrix mean) {
        JMatrix var = new JMatrix(1, featureSize, 1, 1);
        double[] varData = var.getMatrix();
        double[] inputData = input.getMatrix();
        double[] meanData = mean.getMatrix();
        
        int batchSize = input.length();
        int channels = input.channels();
        int height = input.height();
        int width = input.width();
        int spatialSize = height * width;
        
        // Parallelize across channels
        IntStream.range(0, channels)
            .parallel()
            .forEach(c -> {
                double sum = 0;
                double meanVal = meanData[c];
                
                for (int n = 0; n < batchSize; n++) {
                    int batchOffset = n * channels * height * width;
                    int channelOffset = c * height * width;
                    
                    for (int h = 0; h < height; h++) {
                        int rowOffset = h * width;
                        
                        for (int w = 0; w < width; w++) {
                            int idx = batchOffset + channelOffset + rowOffset + w;
                            double diff = inputData[idx] - meanVal;
                            sum += diff * diff;
                        }
                    }
                }
                varData[c] = sum / (batchSize * spatialSize);
            });
        
        return var;
    }
    
    private JMatrix normalize(JMatrix input, JMatrix mean, JMatrix variance) {
        JMatrix normalized = new JMatrix(input.length(), input.channels(), input.height(), input.width());
        double[] normalizedData = normalized.getMatrix();
        double[] inputData = input.getMatrix();
        double[] meanData = mean.getMatrix();
        double[] varData = variance.getMatrix();
        
        int batchSize = input.length();
        int channels = input.channels();
        int height = input.height();
        int width = input.width();
        
        // Pre-compute standard deviation inverses for efficiency
        double[] stdInvs = new double[channels];
        for (int c = 0; c < channels; c++) {
            stdInvs[c] = 1.0 / Math.sqrt(varData[c] + epsilon);
        }
        
        IntStream.range(0, batchSize * channels)
            .parallel()
            .forEach(nc -> {
                int n = nc / channels;
                int c = nc % channels;
                
                double meanVal = meanData[c];
                double stdInv = stdInvs[c];
                int batchOffset = n * channels * height * width;
                int channelOffset = c * height * width;
                
                for (int h = 0; h < height; h++) {
                    int rowOffset = h * width;
                    
                    for (int w = 0; w < width; w++) {
                        int idx = batchOffset + channelOffset + rowOffset + w;
                        normalizedData[idx] = (inputData[idx] - meanVal) * stdInv;
                    }
                }
            });
        
        return normalized;
    }
    
    private JMatrix scaleAndShift(JMatrix normalized, JMatrix gamma, JMatrix beta) {
        JMatrix output = new JMatrix(normalized.length(), normalized.channels(), normalized.height(), normalized.width());
        double[] outputData = output.getMatrix();
        double[] normalizedData = normalized.getMatrix();
        double[] gammaData = gamma.getMatrix();
        double[] betaData = beta.getMatrix();
        
        int batchSize = normalized.length();
        int channels = normalized.channels();
        int height = normalized.height();
        int width = normalized.width();
        
        // Parallelize across batch and channels
        IntStream.range(0, batchSize * channels)
            .parallel()
            .forEach(nc -> {
                int n = nc / channels;
                int c = nc % channels;
                
                double gammaVal = gammaData[c];
                double betaVal = betaData[c];
                int batchOffset = n * channels * height * width;
                int channelOffset = c * height * width;
                
                for (int h = 0; h < height; h++) {
                    int rowOffset = h * width;
                    
                    for (int w = 0; w < width; w++) {
                        int idx = batchOffset + channelOffset + rowOffset + w;
                        outputData[idx] = normalizedData[idx] * gammaVal + betaVal;
                    }
                }
            });
        
        return output;
    }

    public JMatrix backward(JMatrix dOut, double learningRate) {
        int batchSize = input.length();
        int channels = input.channels();
        int height = input.height();
        int width = input.width();
        int spatialSize = height * width;
        int elements = batchSize * spatialSize;
        
        // Reset accumulated gradients
        dGamma.fill(0);
        dBeta.fill(0);
        dxHatSum.fill(0);
        dxHatXhatSum.fill(0);
        
        double[] dGammaData = dGamma.getMatrix();
        double[] dBetaData = dBeta.getMatrix();
        double[] dOutData = dOut.getMatrix();
        double[] xHatData = xHat.getMatrix();
        double[] dxHatSumData = dxHatSum.getMatrix();
        double[] dxHatXhatSumData = dxHatXhatSum.getMatrix();
        double[] dxData = dx.getMatrix();
        double[] dxHatData = dxHat.getMatrix();
        double[] gammaData = gamma.getMatrix();
        double[] varData = batchVar.getMatrix();
        
        // Compute dGamma and dBeta
        IntStream.range(0, channels)
            .parallel()
            .forEach(c -> {
                double dGammaSum = 0;
                double dBetaSum = 0;
                
                for (int n = 0; n < batchSize; n++) {
                    int batchOffset = n * channels * height * width;
                    int channelOffset = c * height * width;
                    
                    for (int h = 0; h < height; h++) {
                        int rowOffset = h * width;
                        
                        for (int w = 0; w < width; w++) {
                            int idx = batchOffset + channelOffset + rowOffset + w;
                            dGammaSum += dOutData[idx] * xHatData[idx];
                            dBetaSum += dOutData[idx];
                        }
                    }
                }
                
                dGammaData[c] = dGammaSum;
                dBetaData[c] = dBetaSum;
            });
        
        // Compute dxHat = dout * gamma
        IntStream.range(0, batchSize * channels)
            .parallel()
            .forEach(nc -> {
                int n = nc / channels;
                int c = nc % channels;
                double gammaVal = gammaData[c];
                
                int batchOffset = n * channels * height * width;
                int channelOffset = c * height * width;
                
                for (int h = 0; h < height; h++) {
                    int rowOffset = h * width;
                    
                    for (int w = 0; w < width; w++) {
                        int idx = batchOffset + channelOffset + rowOffset + w;
                        dxHatData[idx] = dOutData[idx] * gammaVal;
                    }
                }
            });
        
        // Compute intermediate sums for dx calculation
        IntStream.range(0, channels)
            .parallel()
            .forEach(c -> {
                double dxHatSumVal = 0;
                double dxHatXhatSumVal = 0;
                
                for (int n = 0; n < batchSize; n++) {
                    int batchOffset = n * channels * height * width;
                    int channelOffset = c * height * width;
                    
                    for (int h = 0; h < height; h++) {
                        int rowOffset = h * width;
                        
                        for (int w = 0; w < width; w++) {
                            int idx = batchOffset + channelOffset + rowOffset + w;
                            dxHatSumVal += dxHatData[idx];
                            dxHatXhatSumVal += dxHatData[idx] * xHatData[idx];
                        }
                    }
                }
                
                dxHatSumData[c] = dxHatSumVal;
                dxHatXhatSumData[c] = dxHatXhatSumVal;
            });
        
        // Step Calculate dx
        double[] stdInvs = new double[channels];
        for (int c = 0; c < channels; c++) {
            stdInvs[c] = 1.0 / Math.sqrt(varData[c] + epsilon);
        }
        
        IntStream.range(0, batchSize * channels)
            .parallel()
            .forEach(nc -> {
                int n = nc / channels;
                int c = nc % channels;
                
                double stdInv = stdInvs[c];
                double dxHatSumVal = dxHatSumData[c] / elements;
                double dxHatXhatSumVal = dxHatXhatSumData[c] / elements;
                
                int batchOffset = n * channels * height * width;
                int channelOffset = c * height * width;
                
                for (int h = 0; h < height; h++) {
                    int rowOffset = h * width;
                    
                    for (int w = 0; w < width; w++) {
                        int idx = batchOffset + channelOffset + rowOffset + w;
                        dxData[idx] = stdInv * (
                            dxHatData[idx] - 
                            dxHatSumVal - 
                            xHatData[idx] * dxHatXhatSumVal
                        );
                    }
                }
            });
        
        // Update parameters with scaling
        double scale = learningRate / (batchSize * spatialSize);
        gamma = gamma.subtract(dGamma.multiply(scale));
        beta = beta.subtract(dBeta.multiply(scale));
        
        return dx;
    }

    @Override
    protected HashMap<String, JMatrix> getWeights() {
        HashMap<String, JMatrix> parameters = new HashMap<>();
        parameters.put("batch_norm_gamma", gamma);
        parameters.put("batch_norm_beta", beta);
        parameters.put("batch_norm_running_mean", runningMean);
        parameters.put("batch_norm_running_var", runningVar);
        return parameters;
    }

    @Override
    protected int[] getOutputShape() {
        if (input != null) {
            return input.shape();
        }
        return null;
    }
    @Override
    protected HashMap<String, Double> advancedStatistics() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'advancedStatistics'");
    }
}