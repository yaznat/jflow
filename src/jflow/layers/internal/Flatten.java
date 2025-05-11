package jflow.layers.internal;


import jflow.data.JMatrix;
import jflow.layers.templates.ShapeAlteringLayer;

public class Flatten extends ShapeAlteringLayer{

    public Flatten() {
        super("flatten");
    }
    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        JMatrix output = input.reshape(input.length(), input.channels() * 
            input.height() * input.width(), 1, 1);
        return trackOutput(output);
    }

    @Override
    public JMatrix backward(JMatrix input) {
        JMatrix gradient = input.reshape(getPreviousLayer().getOutputShape());
        return trackGradient(gradient);
    }
    
    @Override
    public int[] getOutputShape() {
        int[] prevOutputShape = getPreviousLayer().getOutputShape();
        int flattenedSize = prevOutputShape[1] * prevOutputShape[2] * prevOutputShape[3];

        return new int[]{-1, flattenedSize};
    }
    
}
