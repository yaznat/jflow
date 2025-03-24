package JFlow.Layers;

import JFlow.JMatrix;

abstract class Activation {
    abstract JMatrix applyActivation(JMatrix input);
    abstract JMatrix applyDActivation(JMatrix Z, JMatrix gradient);

}




