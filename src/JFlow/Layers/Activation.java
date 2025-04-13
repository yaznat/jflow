package JFlow.Layers;

import JFlow.JMatrix;

abstract class Activation extends Component{
    private int[] outputShape;

    public Activation(String name, int numTrainableParameters) {
        super(name, 0);
    }
    abstract JMatrix applyActivation(JMatrix input);
    abstract JMatrix applyDActivation(JMatrix Z, JMatrix gradient);

    protected void setOutputShape(int[] outputShape) {
        this.outputShape = outputShape;
    }
    protected int[] getOutputShape() {
        return outputShape;
    }

}




