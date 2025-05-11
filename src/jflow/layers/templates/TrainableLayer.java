package jflow.layers.templates;

import jflow.data.JMatrix;

public abstract class TrainableLayer extends ShapeAlteringLayer{
    public TrainableLayer(String type) {
        super(type);
    }

    public abstract JMatrix[] getParameterGradients();

    public abstract void updateParameters(JMatrix[] paramterUpdates);

    public abstract JMatrix[] getWeights();

    @Override 
    public JMatrix[] debugData() {
        JMatrix[] parameterGradients = getParameterGradients();
        int numParameterGradients = parameterGradients.length;

        JMatrix[] debugData = new JMatrix[numParameterGradients + 1];

        for (int i = 0; i < numParameterGradients; i++) {
            debugData[i] = parameterGradients[i];
        }
        // Ensure dX is properly named
        debugData[numParameterGradients] = getGradient().setName("dX");

        return debugData;
    }
}
