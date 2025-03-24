package JFlow;
import java.util.stream.IntStream;

public class JMatrix {
    private double[] matrix;
    private int length, channels, height, width;

    public JMatrix(int length, int channels, int height, int width) {
        matrix = new double[length * channels * height * width];
        this.length = length;
        this.channels = channels;
        this.height = height;
        this.width = width;
    }

    public JMatrix(double[] matrix, int length, int channels, int height, int width) {
        this.matrix = matrix;
        this.length = length;
        this.channels = channels;
        this.height = height;
        this.width = width;
    }

    public double[] getMatrix() {
        return matrix;
    }

    public void setMatrix(double[] matrix) {
        if (matrix.length != size()) {
            throw new IllegalArgumentException(
                "Sizes must match. Original: " 
                + size() + " New: " + matrix.length
            );
        }
        this.matrix = matrix;
    }

    public int size() {
        return matrix.length;
    }

    public int length() {
        return length;
    }
    public int channels() {
        return channels;
    }
    public int height() {
        return height;
    }
    public int width () {
        return width;
    }
    public void setShape(int[] shape) {
        int newLength = shape[0];
        int newChannels = shape[1];
        int newHeight = shape[2];
        int newWidth = shape[3];

        int numItems = size();
        int newNumItems = newLength * newChannels * newHeight * newWidth;

        if (numItems != newNumItems) {
            throw new IllegalArgumentException(
                "Invalid reshape: total elements must match. Original: " 
                + numItems + " Reshape: " + newNumItems);
        }

        length = newLength;
        channels = newChannels;
        height = newHeight;
        width = newWidth;
    }
    public double max() {
        double max = Double.NEGATIVE_INFINITY;
        for (double d : matrix) {
            max = Math.max(max, d);
        }
        return max;
    }
    public double absMax() {
        double max = Double.NEGATIVE_INFINITY;
        for (double d : matrix) {
            max = Math.max(max, Math.abs(d));
        }
        return max;
    }
    public double mean() {
        double mean = 0;
        for (double d : matrix) {
            mean += d;
        }
        return mean / size();
    }
    public double sum() {
        double sum = 0;
        for (double d : matrix) {
            sum += d;
        }
        return sum;
    }
    public double frobeniusNorm() {
        double sum = 0;
        for (double d : matrix) {
            sum += d * d;
        }
        return Math.sqrt(sum);
    }
    // Sum along rows for flat use cases
    public double[] sum0(boolean scale) {
        int rows = length;
        int cols = channels * height * width;

        double scaleFactor = (scale) ? 1.0 / rows : 1;

        double[] sum = new double[rows];
        IntStream.range(0, rows).parallel().forEach(i -> { 
            for (int j = 0; j < cols; j++) {
                sum[i] = matrix[i * cols + j];
            }
            sum[i] *= scaleFactor;
        });
        return sum;
    }
    // Rotate by 90 degrees for flat use cases
    public JMatrix transpose2D() {
        int oldHeight = length;
        int oldWidth = channels * height * width;
        int newHeight = oldWidth;
        int newWidth = oldHeight;

        double[] rotated = new double[size()];

        IntStream.range(0, oldHeight).parallel().forEach(row -> {
            for (int col = 0; col < oldWidth; col++) {
                int oldIndex = row * oldWidth + col;
                int newIndex = col * newWidth + row;
                rotated[newIndex] = matrix[oldIndex];
            }
        });


        // Assign all features to channels for simplicity
        return new JMatrix(rotated, newHeight, newWidth, 1, 1);
    }
    public void clip(double min, double max) {
        
    }

    // Perform matrix multiplication with another matrix
    public JMatrix dot(JMatrix secondMatrix, boolean scale) {
        // Treat channels * height * width as flat
        int m = length;
        int n = secondMatrix.channels() * secondMatrix.height() * secondMatrix.width();
        int k = channels * height * width;

        if (k != secondMatrix.length()) {
            throw new IllegalArgumentException(
                "Matrix muliplication not possible for" + 
                "arrays with shape: (" + m + "," + k + 
                ") and (" + secondMatrix.length() + "," + 
                n + ")"
            );
        }

        double scaleFactor = 1.0 / Math.sqrt(k);

        double[] matrix1 = matrix;
        double[] matrix2 = secondMatrix.getMatrix();
        double[] result = new double[m * n];

        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int kIndex = 0; kIndex < k; kIndex++) {
                    sum += matrix1[i * k + kIndex] * matrix2[kIndex * n + j];
                }
                if (scale) {
                    result[i * n + j] = sum * scaleFactor;
                } else {
                    result[i * n + j] = sum;
                }
            }
        });

        return new JMatrix(result, m, secondMatrix.channels(), secondMatrix.height(), secondMatrix.width());
    }

    // Subtract another matrix from this matrix
    public JMatrix subtract(JMatrix secondMatrix) {
        // Check for unequal number of elements
        if (size() != secondMatrix.size()) {
            throw new IllegalArgumentException(
                "Sizes " + size() + " and " + 
                secondMatrix.size() + 
                " cannot be broadcast together."
            );
        }
        // Otherwise, assume proper feature lineup
        int size = size();
        double[] matrix1 = matrix;
        double[] matrix2 = secondMatrix.getMatrix();
        double[] result = new double[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = matrix1[i] - matrix2[i];
        });

        return new JMatrix(result, length, channels, height, width);
    }

    // Add another matrix to this matrix
    public JMatrix add(JMatrix secondMatrix) {
        // Check for unequal number of elements
        if (size() != secondMatrix.size()) {
            throw new IllegalArgumentException(
                "Sizes " + size() + " and " + 
                secondMatrix.size() + 
                " cannot be broadcast together."
            );
        }
        // Otherwise, assume proper feature lineup
        int size = size();
        double[] matrix1 = matrix;
        double[] matrix2 = secondMatrix.getMatrix();
        double[] result = new double[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = matrix1[i] + matrix2[i];
        });

        return new JMatrix(result, length, channels, height, width);
    }

    // Multiply another matrix with this matrix
    public JMatrix multiply(JMatrix secondMatrix) {
        // Check for unequal number of elements
        if (size() != secondMatrix.size()) {
            throw new IllegalArgumentException(
                "Sizes " + size() + " and " + 
                secondMatrix.size() + 
                " cannot be broadcast together."
            );
        }
        // Otherwise, assume proper feature lineup
        int size = size();
        double[] matrix1 = matrix;
        double[] matrix2 = secondMatrix.getMatrix();
        double[] result = new double[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = matrix1[i] * matrix2[i];
        });

        return new JMatrix(result, length, channels, height, width);
    }

    // Subtract a scalar from this matrix
    public JMatrix subtract(double scalar) {
        int size = size();
        double[] result = new double[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = matrix[i] - scalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }

    // Add a scalar to this matrix
    public JMatrix add(double scalar) {
        int size = size();
        double[] result = new double[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = matrix[i] + scalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }

    // Multiply a scalar with this matrix
    public JMatrix multiply(double scalar) {
        int size = size();
        double[] result = new double[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = matrix[i] * scalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }
}
