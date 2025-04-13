package JFlow.Layers;

import java.util.HashMap;

import JFlow.JMatrix;

abstract class Component {
    private String name;
    private int numTrainableParameters;
    public Component(String name, int numTrainableParameters){
        this.name = name;
        this.numTrainableParameters = numTrainableParameters;
    }
    protected int numTrainableParameters() {
        return numTrainableParameters;
    }
    protected void setNumTrainableParameters(int numTrainableParameters) {
        this.numTrainableParameters = numTrainableParameters;
    }
    protected String getName() {
        return name;
    }

    protected abstract int[] getOutputShape();

    protected abstract HashMap<String, JMatrix> getWeights();
    protected abstract HashMap<String, Double> advancedStatistics();
}
