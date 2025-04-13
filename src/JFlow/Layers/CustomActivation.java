package JFlow.Layers;

import java.util.HashMap;
import java.util.function.BiFunction;
import java.util.function.Function;

import JFlow.JMatrix;

public class CustomActivation extends Activation{
    private Function<JMatrix, JMatrix> activation;
    private BiFunction<JMatrix, JMatrix, JMatrix> dActivation;



    public CustomActivation(Function<JMatrix, JMatrix> activation, BiFunction<JMatrix, JMatrix, JMatrix> dActivation) {
        super("custom_activation", 0);
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
    @Override
    protected HashMap<String, JMatrix> getWeights() {
        return new HashMap<>();
    }
    @Override
    protected HashMap<String, Double> advancedStatistics() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'advancedStatistics'");
    }
    
    
}
