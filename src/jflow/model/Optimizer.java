package jflow.model;

import java.util.HashMap;
import java.util.Map;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public abstract class Optimizer {
    private HashMap<TrainableLayer, JMatrix[]> layerMoments = new HashMap<>();
    private HashMap<String, TrainableLayer> layerID = new HashMap<>();
    private String name;

    protected Optimizer(String name){
        this.name = name;
    }
    

    public abstract void apply(HashMap<String, JMatrix[]> layerGradients);

    protected abstract void initializeLayer(TrainableLayer layer);

    protected String getName() {
        return name;
    }

    protected HashMap<TrainableLayer, JMatrix[]> getMoments() {
        return layerMoments;
    }
    protected HashMap<String, TrainableLayer> getLayerID() {
        return layerID;
    }

    protected JMatrix[] getWeights() {
        int totalNumWeights = 0;
        // Count number of weights
        for (Map.Entry<TrainableLayer, JMatrix[]> entry : layerMoments.entrySet()) {
            totalNumWeights += entry.getValue().length;
        }
        // Assemble values into an array
        JMatrix[] weights = new JMatrix[totalNumWeights];
        int index = 0;
        for (Map.Entry<TrainableLayer, JMatrix[]> entry : layerMoments.entrySet()) {
            for (JMatrix weight : entry.getValue()) {
                weight.setName(String.valueOf(index)); // Ensure that each weight has a unique name
                weights[index++] = weight;
            }
        }
        return weights;
    }
}
