package JFlow.Layers;

import java.util.function.BiFunction;
import java.util.function.Function;

import JFlow.JMatrix;

public class CustomActivation extends Activation{
    private Function<JMatrix, JMatrix> activation;
    private BiFunction<JMatrix, JMatrix, JMatrix> dActivation;



    public CustomActivation(Function<JMatrix, JMatrix> activation, BiFunction<JMatrix, JMatrix, JMatrix> dActivation) {
        this.activation = activation;
        this.dActivation = dActivation;
    }
    @Override
    JMatrix applyActivation(JMatrix input) {
        return activation.apply(input);
    }

    @Override
    JMatrix applyDActivation(JMatrix Z, JMatrix gradient) {
        return dActivation.apply(Z, gradient);
    }
    
}
