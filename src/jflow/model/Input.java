package jflow.model;

import jflow.data.JMatrix;

public class Input {
    private JMatrix data = null;
    protected Input(){}

    protected void setData(JMatrix data) {
        if (this.data == null) {
            this.data = data;
        } else {
            // Avoid reassigning reference
            this.data.setMatrix(data.getMatrix(), data.shape());
        }
    }
    
    protected JMatrix getData() {
        return data;
    }
}
