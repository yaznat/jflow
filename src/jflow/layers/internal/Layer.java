package jflow.layers.internal;


import jflow.data.JMatrix;

public abstract class Layer {
    private Layer nextLayer;
    private Layer previousLayer;

    private JMatrix output;
    private JMatrix gradient;

    private int[] inputShape;

    private String type;

    private int numTrainableParameters;
    private int IDnum;

    private boolean isShapeInfluencer;
        
    public Layer(String type, boolean isShapeInfluencer) {
        this.type = type;
        this.isShapeInfluencer = isShapeInfluencer;
    }

    public void build() {
        this.IDnum = getLayerTypeCount(type);
    }


    public abstract JMatrix forward(JMatrix input, boolean training);

    public abstract JMatrix backward(JMatrix input);

    public abstract int[] getOutputShape();

    protected abstract JMatrix[] debugData();

    public JMatrix getOutput() {
        return output;
    }
    public JMatrix getGradient() {
        return gradient;
    }

    protected JMatrix trackOutput(JMatrix output) {
        this.output = output;
        return output;
    }
    
    protected JMatrix trackGradient(JMatrix gradient) {
        this.gradient = gradient;
        return gradient;
    }

    public void setInputShape(int[] inputShape) {
        this.inputShape = inputShape;
    }
    
    protected int[] internalGetInputShape() {
        return inputShape;
    }

    public int[] getInputShape() {
        if (internalGetInputShape() != null) {
            return internalGetInputShape();
        }
        return getPreviousLayer().getOutputShape();
    }

    // True if this layer changes output shape (e.g., Dense, Conv2D, Flatten)
    public boolean isShapeInfluencer() {
        return isShapeInfluencer;
    }

    public Layer getPreviousShapeInfluencer() {
        Layer prevLayer = getPreviousLayer();
        while (!prevLayer.isShapeInfluencer()) {
            prevLayer = prevLayer.getPreviousLayer();
        }
        return prevLayer;
    }
    public Layer getNextLayer() {
        return nextLayer;
    }
    public Layer getPreviousLayer() {
        return previousLayer;
    }

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
    }

    public boolean trainable() {
        return numTrainableParameters() != 0;
    }

    protected void setIDNum(int IDnum) {
        this.IDnum = IDnum;
    }
    public int numTrainableParameters() {
        return numTrainableParameters;
    }
    // For proper layer build after initialization
    protected void setNumTrainableParameters(int numTrainableParameters) {
        this.numTrainableParameters = numTrainableParameters;
    }

    public String getType() {
        return type;
    }
    public String getName() {
        return type + "_" + IDnum;
    }

    public int getLayerIndex() {
        if (getPreviousLayer() == null) {
            return 0;
        } else {
            return 1 + getPreviousLayer().getLayerIndex();
        }
    }


    // Count the number of layers in the linked list of a certain type.
    protected int getLayerTypeCount(String layerType) {
        int count = 1;
        Layer prevLayer = getPreviousLayer();
        while (prevLayer != null) {
            if (prevLayer.getType().equals(layerType)) {
                count++;
            }
            prevLayer = prevLayer.getPreviousLayer();
        }
        return count;
    }

    public void printDebug() {
        JMatrix[] debugData = debugData();
        String title = getName() + " gradients";
        System.out.println("\033[94m╭──────────────────── \033[0m\033[1;94m" + 
        title + "\033[0m\033[94m ────────────────────╮\033[0m");

        // Iterate over entries
        for (JMatrix data : debugData) {
            String dataName = data.getName();
            System.out.print("\033[94m│\033[0m " + "\033[38;2;0;153;153m" + dataName + "\033[0m");
            // Print statistics for the entry
            System.out.print(" \033[38;2;222;197;15m|\033[0m \033[38;2;255;165;0mshape:\033[0m \033[37m" + data.shapeAsString() + "\033[0m"); // shape
            System.out.print(" \033[38;2;222;197;15m|\033[0m \033[38;2;255;165;0mabsmax:\033[0m \033[37m" + data.absMax() + "\033[0m"); // absmax
            System.out.print(" \033[38;2;222;197;15m|\033[0m \033[38;2;255;165;0mmean:\033[0m \033[37m" + data.mean() + "\033[0m"); // mean
            System.out.print(" \033[38;2;222;197;15m|\033[0m \033[38;2;255;165;0mL1:\033[0m \033[37m" + data.l1Norm() + "\033[0m\n"); // L1 norm
        }
        String closer = "\033[94m╰─────────────────────";
        for (int i = 0; i < title.length(); i++) {
            closer += "─";
        }
        closer += "─────────────────────╯\033[0m";
        System.out.println(closer);
    }
}
