package jflow.model;

/**
 * A layer in a JFlow Sequential model
 */
public class Layer {
    private jflow.layers.internal.Layer layer;

    protected Layer(jflow.layers.internal.Layer layer) {
        this.layer = layer;
    }

    protected jflow.layers.internal.Layer getInternal() {
        return layer;
    }

    
}
