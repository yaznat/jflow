package JFlow.Layers;

import java.util.HashMap;

import JFlow.JMatrix;

class Reshape extends Layer{
    private int newChannels, newHeight, newWidth, oldChannels, oldHeight, oldWidth, oldLength;

    public Reshape(int channels, int height, int width) {
        super("reshape", 0);
        this.newChannels = channels;
        this.newHeight = height;
        this.newWidth = width;
    }

    @Override
    protected int channels() {
        return newChannels;
    }

    @Override
    public void forward(JMatrix input, boolean training) {
        this.oldLength = input.length();
        this.oldChannels = input.channels();
        this.oldHeight = input.height();
        this.oldWidth = input.width();


        // Account for dense layers being transposed
        if (getPreviousLayer() instanceof Dense) {
            input = input.transpose2D();
        }

         // Improper dimension catching is handled inside of JMatrix.reshape()
        if (getNextLayer() != null) {
            getNextLayer().forward(input.reshape(input.length(), newChannels, newHeight, newWidth), training);
        }
    }

    

    @Override
    public void backward(JMatrix input, double learningRate) {
        int batchSize = input.length();
        int channels = input.channels();
        int height = input.height();
        int width = input.width();
        
        if (super.getDebug()) {
            System.out.println("reshape");
            System.out.println("Input images:" + batchSize);
            System.out.println("Input channels:" + channels);
            System.out.println("Input height:" + height);
            System.out.println("Input width:" + width);
        }

        // Improper dimension catching is handled inside of JMatrix.reshape()
        if (getPreviousLayer() != null) {
            getPreviousLayer().backward(input.reshape(oldLength, oldChannels, oldHeight, oldWidth), learningRate);
        }
    }

    @Override
    public JMatrix getOutput() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getOutput'");
    }

    @Override
    public JMatrix getGradient() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getGradient'");
    }

    @Override
    protected HashMap<String, JMatrix> getWeights() {
        return null;
    }

    @Override
    protected int[] getOutputShape() {
        return new int[] {oldLength, newChannels, newHeight, newWidth};
    }

    @Override
    protected HashMap<String, Double> advancedStatistics() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'advancedStatistics'");
    }
    
}
