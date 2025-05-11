package jflow.model;

public class InputShape {
    private int flattenedSize, channels, height, width;
    private boolean flat;

    protected InputShape(int flattenedSize) {
        this.flattenedSize = flattenedSize;
        this.flat = true;
    }

    protected InputShape(int channels, int height, int width) {
        this.channels = channels;
        this.height = height;
        this.width = width;
        this.flat = false;
    }

    protected int[] getShape() {
        if (flat) {
            return new int[]{flattenedSize};
        }
        return new int[]{channels, height, width};
    }
}
