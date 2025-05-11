package jflow.layers.internal;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapePreservingLayer;

public class ReLU extends ShapePreservingLayer{
    public ReLU(){
        super("re_lu");
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        int size = input.size();
        JMatrix Z = input.zerosLike();

        IntStream.range(0, size).parallel().forEach(i -> {
            Z.set(i, (input.get(i) > 0) ? input.get(i) : 0);
        });

        return trackOutput(Z);
    }

    @Override
    public JMatrix backward(JMatrix gradient) {
        int size = gradient.size();
        JMatrix output = getOutput();
        JMatrix dZ = output.zerosLike();

        IntStream.range(0, size).parallel().forEach(i -> {
            dZ.set(i, (output.get(i) > 0) ? gradient.get(i) : 0);
        });
       
        return trackGradient(dZ);
    }
}

