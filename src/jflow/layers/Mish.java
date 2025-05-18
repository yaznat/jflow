package jflow.layers;

import java.util.stream.IntStream;
import jflow.data.JMatrix;
import jflow.layers.templates.ShapePreservingLayer;

public class Mish extends ShapePreservingLayer {
    private JMatrix lastInput;
    
    public Mish() {
        super("mish");
    }
    
    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        if (training) {
            lastInput = input;
        }
        int size = input.size();
        JMatrix output = input.zerosLike();
        
        IntStream.range(0, size).parallel().forEach(i -> {
            double x = input.get(i);
            double softplus = Math.log1p(Math.exp(x));
            output.set(i, x * Math.tanh(softplus));
        });
        
        return trackOutput(output, training);
    }
    
    @Override
    public JMatrix backward(JMatrix gradient) {
        int size = gradient.size();
        JMatrix input = lastInput; // Use original input x
        JMatrix dZ = input.zerosLike();
        
        IntStream.range(0, size).parallel().forEach(i -> {
            double x = input.get(i);
            double ex = Math.exp(x);
            double softplus = Math.log1p(ex);
            double tanhSoftplus = Math.tanh(softplus);
            
            // Derivative of Mish: x * d(tanh(softplus(x)))/dx + tanh(softplus(x))
            // where d(tanh(softplus(x)))/dx = (1 - tanh(softplus(x))^2) * ex/(1 + ex)
            double derivative = tanhSoftplus + x * (1 - tanhSoftplus * tanhSoftplus) * ex / (1 + ex);
            
            dZ.set(i, gradient.get(i) * derivative);
        });
        
        return trackGradient(dZ);
    }
}
