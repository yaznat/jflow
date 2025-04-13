package JFlow;
import JFlow.data.Image;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;


public class JMatrix {
    private double[] matrix;
    private int length, channels, height, width;
    private Random rand = new Random();
    private ArrayList<Image> imageList;
    boolean isImageWrapper;

    // Create a new, empty JMatrix
    public JMatrix(int length, int channels, int height, int width) {
        matrix = new double[length * channels * height * width];
        this.length = length;
        this.channels = channels;
        this.height = height;
        this.width = width;
    }

    // Wrap a double[] in a JMatrix
    public JMatrix(double[] matrix, int length, int channels, int height, int width) {
        this.matrix = matrix;
        this.length = length;
        this.channels = channels;
        this.height = height;
        this.width = width;
    }

    // Wrap image objects in a JMatrix
    public JMatrix(ArrayList<Image> imageList) {
        this.imageList = imageList;
        this.length = imageList.size();
        this.channels = imageList.getFirst().numChannels();
        this.height = imageList.getFirst().getHeight();
        this.width = imageList.getFirst().getWidth();
        isImageWrapper = true;

    }

    protected double access(int index) {
        if (imageList == null) {
            return matrix[index];
        } else {
            int imageSize = channels * height * width;
            int lengthIndex = length / index;
            Image access = imageList.get(lengthIndex);
            int pixel = index % imageSize;
            return access.getPixel(pixel);
        }
    }


    public double[] getMatrix() {
        return matrix;
    }

    public int[] shape() {
        return new int[]{length, channels, height, width};
    }
    public void printShape() {
        System.out.println("(" + length + "," + channels + 
            "," + height + "," + width + ")");
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
    // Get a channels * height * width element
    public double[] get(int lengthIndex) {
        int sliceSize = channels * height * width;
        int startIdx = lengthIndex * sliceSize;  
        double[] slice = new double[sliceSize];
        System.arraycopy(matrix, startIdx, slice, 0, sliceSize);
        return slice;
    }

     // Get a channels * height * width element as a JMatrix
     public JMatrix getWrapped(int lengthIndex) {
        int sliceSize = channels * height * width;
        int startIdx = lengthIndex * sliceSize;  
        double[] slice = new double[sliceSize];
        System.arraycopy(matrix, startIdx, slice, 0, sliceSize);
        return new JMatrix(slice, 1, channels, height, width);
    }

    // Get a height * width element
    public double[] get(int lengthIndex, int channelIndex) {
        int sliceSize = height * width;
        int startIdx = lengthIndex * channels * sliceSize + channelIndex * sliceSize;
        double[] slice = new double[sliceSize];
        System.arraycopy(matrix, startIdx, slice, 0, sliceSize);
        return slice;
    }

    // Get a height * width element as a JMatrix
    public JMatrix getWrapped(int lengthIndex, int channelIndex) {
        int sliceSize = height * width;
        int startIdx = lengthIndex * channels * sliceSize + channelIndex * sliceSize;
        double[] slice = new double[sliceSize];
        System.arraycopy(matrix, startIdx, slice, 0, sliceSize);
        return new JMatrix(slice, 1, 1, height, width);
    }

    // Set a height * width slice
    public void set(int lengthIndex, int channelIndex, double[] values) {
        int sliceSize = height * width;
        if (values.length != sliceSize) {
            throw new IllegalArgumentException("Invalid slice size. Expected " + sliceSize + " values.");
        }
        int startIdx = lengthIndex * channels * sliceSize + channelIndex * sliceSize;
        System.arraycopy(values, 0, matrix, startIdx, sliceSize);
    }

    
    
    public JMatrix reshape(int newLength, int newChannels, int newHeight, int newWidth) {
        int numItems = size();
        int newNumItems = newLength * newChannels * newHeight * newWidth;

        if (numItems != newNumItems) {
            throw new IllegalArgumentException(
                "Invalid reshape: total elements must match. Original: " 
                + numItems + " Reshape: " + newNumItems);
        }

        return new JMatrix(matrix, newLength, newChannels, newHeight, newWidth);
    }
    // Set an item with 1D indexing
    public void set(int index, double val) {
        matrix[index] = val;
    }

// Statistics
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
    // Return the 1D index of the maximum value
    public int argmax() {
        double maxValue = Double.NEGATIVE_INFINITY;
        int maxIndex = 0;
        for (int i = 0; i < size(); i++) {
            if (access(i) > maxValue) {
                maxValue = access(i);
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    // Return the index of the maximum value based on axis
    public int[] argmax(int axis) {
        int[] positions = new int[0];
        int itemSize = 0; 
        int numRows = 0;
        switch (axis) {
            case 0: 
                // argmax of each channel * height * width
                itemSize = channels * height * width;
                numRows = length;
                positions = new int[numRows];
                break;
            case 1:
                // argmax of each height * width
                itemSize = height * width;
                numRows = length * channels;
                positions = new int[numRows];
                break;
            case 2:
                // argmax of each width
                itemSize = width;
                numRows = length * channels * height;
                positions = new int[numRows];
                break;
            default:
                throw new IllegalArgumentException("Invalid axis: " + axis);
        }
        // Find max positions
        for (int i = 0; i < numRows; i++) {
            double maxValue = Double.NEGATIVE_INFINITY;
            int maxIndex = 0;
            for (int j = 0; j < itemSize; j++) {
                if (access(i * itemSize + j) > maxValue) {
                    maxValue = access(i * itemSize + j);
                    maxIndex = j;
                }
            }
            positions[i] = maxIndex;
        }
        return positions;
    }
    public double mean() {
        double mean = 0;

        int size = size();
        for (int i = 0; i < size; i++) {
            mean += access(i);
        
        }
        return mean / size;
    }
    public double sum() {
        double sum = 0;

        int size = size();
        for (int i = 0; i < size; i++) {
            sum += access(i);
        
        }
        return sum;
    }
   
    public double frobeniusNorm() {
        double sum = 0;

        int size = size();
        for (int i = 0; i < size; i++) {
            double pixel = access(i);
            sum += pixel * pixel;
        
        }
        return Math.sqrt(sum);
    }

    // Scale values from [0,n] to [0,1]
    public JMatrix scaleSigmoid() {
        double max = max();
        return multiply(1.0 / max);
    }

    // Add Gaussian noise to each item
    public JMatrix addGaussianNoise(double mean, double stdDev) {
        double[] noisy = new double[size()];
        for (int i = 0; i < size(); i++) {
            noisy[i] = access(i) + rand.nextGaussian() * stdDev + mean;
        }
        return new JMatrix(noisy, length, channels, height, width);
    }
    // Sum along rows for 2D use cases
    public double[] sum0(boolean scale) {
        int rows = length;
        int cols = channels * height * width;

        double scaleFactor = (scale) ? 1.0 / rows : 1;

        double[] sum = new double[rows];
        IntStream.range(0, rows).parallel().forEach(i -> { 
            for (int j = 0; j < cols; j++) {
                sum[i] = access(i * cols + j);
            }
            sum[i] *= scaleFactor;
        });
        return sum;
    }
    // Rotate by 90 degrees for 2D use cases
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
                rotated[newIndex] = access(oldIndex);
            }
        });


        // Assign all features to channels for simplicity
        return new JMatrix(rotated, newHeight, newWidth, 1, 1);
    }
    // Compare shape to another JMatrix
    public boolean isSameShapeAs(JMatrix other) {
        return length == other.length() && channels == other.channels()
            && height == other.height() && width == other.width();
    }

    // Copy values
    public JMatrix copy() {
        return new JMatrix(matrix.clone(), length, channels, height, width);
    }
    // Return a new empty JMatrix with the same dimensions
    public JMatrix zerosLike() {
        return new JMatrix(new double[size()], length, channels, height, width);
    }

    public void clip(double min, double max) {
        Arrays.parallelSetAll(matrix, i -> Math.max(min, Math.min(max, matrix[i])));
    }

    public void fill(double fillValue) {
        Arrays.fill(matrix, fillValue);
    }

    // // Perform matrix multiplication with another matrix for 2D use cases
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

        double[] result = new double[m * n];

        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int kIndex = 0; kIndex < k; kIndex++) {
                    sum += access(i * k + kIndex) * secondMatrix.access(kIndex * n + j);
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
    // Take the inverse of each item
    public JMatrix reciprocal() {
        int size = size();
        double[] result = new double[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = 1.0 / access(i);
        });

        return new JMatrix(result, length, channels, height, width);
    }

    // Broadcast sqrt(x) across the matrix
    public JMatrix sqrt() {
        int size = size();
        double[] result = new double[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = Math.sqrt(access(i));
        });

        return new JMatrix(result, length, channels, height, width);
    }
 
    // Subtract another matrix from this matrix
    public JMatrix subtract(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            double[] result = new double[size];
            IntStream.range(0, size).parallel().forEach(i -> {
                result[i] = access(i) - secondMatrix.access(i);
            });
            return new JMatrix(result, length, channels, height, width);
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            double[] result = new double[size];
            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    double subtractor = secondMatrix.access(c); // one subtractor per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        result[offset + i] = access(offset + i) - subtractor;
                    }
                }
            });

            return new JMatrix(result, length, channels, height, width);
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match, (length,1,1,1), or (1,channels,1,1)."
        );
    }


     // Add another matrix to this matrix
     public JMatrix add(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise addition
        if (size == secondMatrix.size()) {
            double[] result = new double[size];
            IntStream.range(0, size).parallel().forEach(i -> {
                result[i] = access(i) + secondMatrix.access(i);
            });
            return new JMatrix(result, length, channels, height, width);
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            double[] result = new double[size];
            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    double adder = secondMatrix.access(c); // one adder per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        result[offset + i] = access(offset + i) + adder;
                    }
                }
            });

            return new JMatrix(result, length, channels, height, width);
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match, (length,1,1,1), or (1,channels,1,1)."
        );
    }

    // Multiply another matrix with this matrix
    public JMatrix multiply(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            double[] result = new double[size];
            IntStream.range(0, size).parallel().forEach(i -> {
                result[i] = access(i) * secondMatrix.access(i);
            });
            return new JMatrix(result, length, channels, height, width);
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            double[] result = new double[size];
            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    double multiplier = secondMatrix.access(c); // one multiplier per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        result[offset + i] = access(offset + i) * multiplier;
                    }
                }
            });

            return new JMatrix(result, length, channels, height, width);
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match, (length,1,1,1), or (1,channels,1,1)."
        );
    }

    // Subtract another matrix from this matrix
    public JMatrix divide(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            double[] result = new double[size];
            IntStream.range(0, size).parallel().forEach(i -> {
                result[i] = access(i) / secondMatrix.access(i);
            });
            return new JMatrix(result, length, channels, height, width);
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            double[] result = new double[size];
            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    double divisor = secondMatrix.access(c); // one divisor per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        result[offset + i] = access(offset + i) / divisor;
                    }
                }
            });

            return new JMatrix(result, length, channels, height, width);
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match, (length,1,1,1), or (1,channels,1,1)."
        );
    }

    // Subtract a scalar from this matrix
    public JMatrix subtract(double scalar) {
        int size = size();
        double[] result = new double[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = access(i) - scalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }

    // Add a scalar to this matrix
    public JMatrix add(double scalar) {
        int size = size();
        double[] result = new double[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = access(i) + scalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }

    // Multiply a scalar with this matrix
    public JMatrix multiply(double scalar) {
        int size = size();
        double[] result = new double[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = access(i) * scalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }

    // Divide this matrix by a scalar
    public JMatrix divide(double scalar) {
        int size = size();
        double[] result = new double[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = access(i) / scalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }
}
