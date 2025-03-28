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


    protected abstract void forward(JMatrix input, boolean training);

    protected abstract void backward(JMatrix input, double learningRate);

    protected abstract JMatrix getOutput();
    protected abstract JMatrix getGradient();

    protected void setDebug(boolean debug) {
        this.debug = debug;
    }

    protected boolean getDebug() {
        return debug;
    }

    protected void setActivation(Activation activation) {
        this.activation = activation;
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

    protected int numTrainableParameters() {
        return numTrainableParameters;
    }
    protected void setIDnum(int IDnum) {
        nameID += "_" + IDnum;
    }
    protected String getNameID() {
        return nameID;
    }

    protected Activation getActivation() {
        return activation;
    }
}
