package jflow.model;

import java.util.HashMap;
import java.util.Map;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public class Adam extends Optimizer{
    private double beta1, beta2, learningRate, epsilon = 1e-8;
    private int timesteps = 0;

    public Adam(double beta1, double beta2, double learningRate) {
        super("adam");
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.learningRate = learningRate;
    }

    public Adam(double learningRate) {
        super("adam");
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.learningRate = learningRate;
    }


    @Override
    public void apply(HashMap<String, JMatrix[]> layerGradients) {
        timesteps++;
        for (Map.Entry<String, JMatrix[]> entry : layerGradients.entrySet()) {
            TrainableLayer layer = getLayerID().get(entry.getKey());
            // System.out.println(entry.getKey());
            JMatrix[] gradients = entry.getValue();

            JMatrix[] moments = getMoments().get(layer);

            JMatrix[] updates = new JMatrix[gradients.length];

            
            for (int i = 0; i < gradients.length; i++) {
                JMatrix weightGradients = gradients[i];
                JMatrix mWeights = moments[2 * i];
                JMatrix vWeights = moments[2 * i + 1];

                // Update first moments (momentum)
                mWeights.multiplyInPlace(beta1).addInPlace(weightGradients.multiply(1 - beta1));

                // Update second moments (velocity)
                vWeights.multiplyInPlace(beta2).addInPlace(weightGradients.multiply(weightGradients).multiply(1 - beta2));

                // Calculate bias-corrected moments
                JMatrix mWeightsCorrected = mWeights.divide(1 - Math.pow(beta1, timesteps));
                JMatrix vWeightsCorrected = vWeights.divide(1 - Math.pow(beta2, timesteps));

                // Calculate parameter updates
                JMatrix weightUpdate = mWeightsCorrected.divideInPlace(
                    vWeightsCorrected.sqrt().addInPlace(epsilon))
                        .multiplyInPlace(learningRate);
                
                updates[i] = weightUpdate;
            }

            // Apply updates
            layer.updateParameters(updates);
        }
    }
}
