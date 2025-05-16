package jflow.model;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapeAlteringLayer;

public abstract class FunctionalLayer extends ShapeAlteringLayer {
    private Layer[] components;
    public FunctionalLayer(String name) {
        super(name);
    }

    @Override
    public void build(int IDnum) {
        super.build(IDnum);
        this.components = defineLayers();
        for (Layer l : components) {
            l.setEnclosingLayer(this);
        }
    }
    
    public abstract Layer[] defineLayers();

    public abstract JMatrix forward(JMatrix input, boolean training);
    public abstract JMatrix backward(JMatrix input);

    public abstract int[] outputShape();

    public Layer[] getLayers() {
        return components;
    }
}
