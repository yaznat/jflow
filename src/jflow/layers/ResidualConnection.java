package jflow.layers;

import jflow.data.JMatrix;
import jflow.model.Layer;

public abstract class ResidualConnection {
    private Layer layer1, layer2;

    public ResidualConnection(Layer layer1, Layer layer2) {
        this.layer1 = layer1;
        this.layer2 = layer2;
    }

    public abstract JMatrix getForwardInput();
    public abstract JMatrix getBackwardInput();

    protected Layer getLayer1() {
        return layer1;
    }

    protected Layer getLayer2() {
        return layer2;
    }
}
