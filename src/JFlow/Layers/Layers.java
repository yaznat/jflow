package JFlow.Layers;

import java.util.function.BiFunction;
import java.util.function.Function;

import JFlow.JMatrix;

public class Layers {
    // Add layers statically for user interface
    public static Layer Dense(int inputSize, int outputSize) {
        return new Dense(inputSize, outputSize);
    }
    public static Layer Conv2D(int numFilters, int filterSize, String padding) {
        return new Conv2D(numFilters, filterSize, padding);
    }
    public static Layer MaxPool2D(int poolSize, int stride) {
        return new MaxPool2D(poolSize, stride);
    }
    public static Activation ReLU() {
        return new ReLU();
    }
    public static Activation LeakyReLU(double alpha) {
        return new LeakyReLU(alpha);
    }

    public static Activation Softmax() {
        return new Softmax();
    }

    public static Dropout Dropout(double alpha) {
        return new Dropout(alpha);
    }

    public static Activation Sigmoid() {
        return new Sigmoid();
    }

    public static Layer Reshape(int channels, int height, int width) {
        return new Reshape(channels, height, width);
    }

    public static Layer Upsampling2D(int scaleFactor) {
        return new Upsampling2D(scaleFactor);
    }

    public static BatchNorm BatchNorm() {
        return new BatchNorm();
    }

    public static Activation customActivation(Function<JMatrix, JMatrix> activation, BiFunction<JMatrix, JMatrix, JMatrix> dActivation) {
        return new CustomActivation(activation, dActivation);
    }

    public static Activation Tanh() {
        return new Tanh();
    }
}
