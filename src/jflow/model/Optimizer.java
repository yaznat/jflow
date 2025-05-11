package jflow.model;

import java.util.HashMap;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public abstract class Optimizer {
    private HashMap<TrainableLayer, JMatrix[]> layerMoments = new HashMap<>();
    private HashMap<String, TrainableLayer> layerID = new HashMap<>();
    private String name;

    protected Optimizer(String name){

    }
    

    public abstract void apply(HashMap<String, JMatrix[]> layerGradients);

    protected void init(TrainableLayer layer) {
        JMatrix[] gradients = layer.getParameterGradients();
        int numWeights = gradients.length;

        JMatrix[] moments = new JMatrix[numWeights * 2];

        for (int i = 0; i < numWeights; i++) {
            // Initialized to zero
            JMatrix mWeights = new JMatrix(gradients[i].shape());
            moments[2 * i] = mWeights;
        
            JMatrix vWeights = new JMatrix(gradients[i].shape());
            moments[2 * i + 1] = vWeights;
        }

        layerMoments.put(layer, moments);
        layerID.put(layer.getName(), layer);
    }

    protected String getName() {
        return name;
    }

    protected HashMap<TrainableLayer, JMatrix[]> getMoments() {
        return layerMoments;
    }
    protected HashMap<String, TrainableLayer> getLayerID() {
        return layerID;
    }


    protected JMatrix stabilizeGradients(JMatrix gradient, int layerIndex) {
        // Create a copy to avoid modifying the original
        JMatrix stabilized = gradient.copy();
        
        // Apply gradient clipping with layer-specific thresholds
        double clipMin = -getClipThreshold(layerIndex);
        double clipMax = getClipThreshold(layerIndex);
        
        // Element-wise clipping
        stabilized.clip(clipMin, clipMax);
        
        // Detect and handle vanishing gradients
        double norm = stabilized.frobeniusNorm();
        
        // Handle vanishing gradients (layer-specific minimum norm threshold)
        double minNormThreshold = getMinNormThreshold(layerIndex);
        if (norm < minNormThreshold && norm > 0) {
            double scaleFactor = minNormThreshold / norm;
            stabilized.multiplyInPlace(scaleFactor);
        }
        
        // Handle exploding gradients (layer-specific maximum norm threshold)
        double maxNormThreshold = getMaxNormThreshold(layerIndex);
        if (norm > maxNormThreshold) {
            double scaleFactor = maxNormThreshold / norm;
            stabilized.multiplyInPlace(scaleFactor);
        }
        
        return stabilized;
    }
    
    /**
     * Calculates layer-specific clipping threshold
     * @param layerIndex The index of the layer
     * @return The clipping threshold
     */
    private double getClipThreshold(int layerIndex) {
        // Base threshold
        double baseThreshold = 1.0;
        
        // Deeper layers may need different thresholds
        // Early layers often need smaller thresholds
        if (layerIndex < 3) {
            return baseThreshold * (0.5 + 0.25 * layerIndex);
        } 
        // Middle layers
        else if (layerIndex < 6) {
            return baseThreshold;
        }
        // Deeper layers may need larger thresholds
        else {
            return baseThreshold * (1.0 + 0.15 * (layerIndex - 5));
        }
    }
    
    /**
     * Calculates minimum gradient norm threshold to prevent vanishing gradients
     * @param layerIndex The index of the layer
     * @return The minimum norm threshold
     */
    private double getMinNormThreshold(int layerIndex) {
        // Base threshold
        double baseThreshold = 1e-6;
        
        // Deeper layers are more prone to vanishing gradients
        return baseThreshold * Math.pow(1.5, layerIndex);
    }
    
    /**
     * Calculates maximum gradient norm threshold to prevent exploding gradients
     * @param layerIndex The index of the layer
     * @return The maximum norm threshold
     */
    private double getMaxNormThreshold(int layerIndex) {
        // Base threshold
        double baseThreshold = 10.0;
        
        // Earlier layers can handle larger gradients
        if (layerIndex < 3) {
            return baseThreshold * 1.5;
        }
        // Middle layers
        else if (layerIndex < 6) {
            return baseThreshold;
        }
        // Deeper layers need more strict control
        else {
            return baseThreshold / (1.0 + 0.2 * (layerIndex - 5));
        }
    }
    
    /**
     * Calculates layer-specific learning rate
     * @param layerIndex The index of the layer
     * @return The adjusted learning rate
     */
    protected double getAdaptiveLearningRate(double learningRate, int layerIndex) {
        // Apply scaling based on layer depth
        // Earlier layers often need smaller learning rates
        if (layerIndex < 3) {
            return learningRate * (0.8 + 0.1 * layerIndex);
        }
        // Middle layers use standard learning rate
        else if (layerIndex < 6) {
            return learningRate;
        }
        // Deeper layers might need smaller learning rates
        else {
            return learningRate / (1.0 + 0.1 * (layerIndex - 5));
        }
    }
}
