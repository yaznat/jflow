package jflow.layers;

import java.util.function.BiFunction;
import java.util.function.Function;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapePreservingLayer;

public class CustomActivation extends ShapePreservingLayer{
    private Function<JMatrix, JMatrix> activation;
    private BiFunction<JMatrix, JMatrix, JMatrix> dActivation;



    public CustomActivation(Function<JMatrix, JMatrix> activation, BiFunction<JMatrix, JMatrix, JMatrix> dActivation, String name) {
        super(name);
        this.activation = activation;
        this.dActivation = dActivation;
    }
    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        JMatrix output = activation.apply(input);

        return trackOutput(output, training);
    }

    @Override
    public JMatrix backward(JMatrix gradient) {
        gradient = dActivation.apply(getOutput(), gradient);

        return trackGradient(gradient);
    }

}
