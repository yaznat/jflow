package JFlow.Layers;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

import JFlow.JMatrix;

abstract class Layer {

    private Layer nextLayer;
    private Layer previousLayer;

    private Activation activation;

    private int numTrainableParameters;
    private String nameID;
    private Dropout dropout;
    private boolean debug;

    protected Layer(int numTrainableParameters, String nameID) {
        this.numTrainableParameters = numTrainableParameters;
        this.nameID = nameID;
    }


    public abstract void forward(JMatrix input, boolean training);

    public abstract void backward(JMatrix input, double learningRate);

    public abstract JMatrix getOutput();
    public abstract JMatrix getGradient();

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public boolean getDebug() {
        return debug;
    }

    public void setActivation(Activation activation) {
        this.activation = activation;
    }

    public Layer getNextLayer() {
        return nextLayer;
    }
    public Layer getPreviousLayer() {
        return previousLayer;
    }

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
    }
    
    public void setDropout(Dropout dropout) {
        this.dropout = dropout;
    }

    public Dropout getDropout() {
        return dropout;
    }

    public int numTrainableParameters() {
        return numTrainableParameters;
    }
    public void setIDnum(int IDnum) {
        nameID += "_" + IDnum;
    }
    public String getNameID() {
        return nameID;
    }

    protected Activation getActivation() {
        return activation;
    }
}
