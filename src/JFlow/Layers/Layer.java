package JFlow.Layers;


import JFlow.JMatrix;

abstract class Layer extends Component{

    private Layer nextLayer;
    private Layer previousLayer;

    private Activation activation;
    private Dropout dropout;
    private BatchNorm batchNorm;

    private boolean debug;

    protected Layer(String name, int numTrainableParameters) {
        super(name, numTrainableParameters);
    }


    protected abstract void forward(JMatrix input, boolean training);

    protected abstract void backward(JMatrix input, double learningRate);

    protected abstract JMatrix getOutput();
    protected abstract JMatrix getGradient();

    protected abstract int channels();

    protected void setDebug(boolean debug) {
        this.debug = debug;
    }

    protected boolean getDebug() {
        return debug;
    }

    protected void setActivation(Activation activation) {
        this.activation = activation;
    }

    protected void setBatchNorm(BatchNorm batchNorm) {
        this.batchNorm = batchNorm;
    }

    protected BatchNorm getBatchNorm() {
        return batchNorm;
    }

    protected Layer getNextLayer() {
        return nextLayer;
    }
    protected Layer getPreviousLayer() {
        return previousLayer;
    }

    protected void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    protected void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
    }
    
    protected void setDropout(Dropout dropout) {
        this.dropout = dropout;
    }

    protected Dropout getDropout() {
        return dropout;
    }

    protected Activation getActivation() {
        return activation;
    }

    
}
