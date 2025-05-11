package jflow.layers.templates;

import jflow.data.JMatrix;
import jflow.layers.internal.Layer;

public abstract class ShapePreservingLayer extends Layer{

    public ShapePreservingLayer(String type) {
        super(type, false);
    }

    public abstract JMatrix forward(JMatrix input, boolean training);

    public abstract JMatrix backward(JMatrix input);

    @Override
    public int[] getOutputShape() {
        return getPreviousLayer().getOutputShape();
    }    

    @Override
    protected JMatrix[] debugData() {
        return new JMatrix[]{getGradient().setName("dX")};
    }
}
