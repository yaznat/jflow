package jflow.data;
import java.util.Random;
import java.util.stream.IntStream;



public class JMatrix {
    private float[] matrix;
    private int length, channels, height, width;
    private Random rand = new Random();
    private String name = null;

    /**
     * Initialize a new JMatrix with default values of zero.
     * @param length                The batch dimension.
     * @param channels              The channel dimension.
     * @param height                The height dimension.
     * @param width                 The width dimension.
     */
    public JMatrix(int length, int channels, int height, int width) {
        matrix = new float[length * channels * height * width];
        this.length = length;
        this.channels = channels;
        this.height = height;
        this.width = width;
    }

     /**
     * Initialize a new JMatrix with default values of zero.
     * @param shape                The desired shape (N, channels, height, width)
     * @throws IllegalArgumentException if the length of shape is not four.
     */
    public JMatrix(int[] shape) {
        this.length = shape[0];
        this.channels = shape[1];
        this.height = shape[2];
        this.width = shape[3];
        if (shape.length != 4) {
            throw new IllegalArgumentException("Invalid shape. Only 4 is permitted.");
        }
        matrix = new float[length * channels * height * width];
    }

    /**
     * Wrap an array in a new JMatrix.
     * @param length                The batch dimension.
     * @param channels              The channel dimension.
     * @param height                The height dimension.
     * @param width                 The width dimension.
     */
    public JMatrix(float[] matrix, int length, int channels, int height, int width) {
        this.matrix = matrix;
        this.length = length;
        this.channels = channels;
        this.height = height;
        this.width = width;
    }

    /**
     * Initialize a new JMatrix with default values of zero.
     * @param length                The batch dimension.
     * @param channels              The channel dimension.
     * @param height                The height dimension.
     * @param width                 The width dimension.
     * @param name                  The name to assign to this JMatrix.
     */
    public JMatrix(int length, int channels, int height, int width, String name) {
        matrix = new float[length * channels * height * width];
        this.length = length;
        this.channels = channels;
        this.height = height;
        this.width = width;
        this.name = name;
    }

     /**
     * Initialize a new JMatrix with default values of zero.
     * @param shape                The desired shape (N, channels, height, width)
     * @param name                  The name to assign to this JMatrix.
     * 
     * @throws IllegalArgumentException if the length of shape is not four.
     */
    public JMatrix(int[] shape, String name) {
        if (shape.length != 4) {
            throw new IllegalArgumentException("Invalid shape. Only 4 is permitted.");
        }

        this.length = shape[0];
        this.channels = shape[1];
        this.height = shape[2];
        this.width = shape[3];
        this.name = name;

        matrix = new float[length * channels * height * width];
    }

    /**
     * Wrap an array in a new JMatrix.
     * @param length                The batch dimension.
     * @param channels              The channel dimension.
     * @param height                The height dimension.
     * @param width                 The width dimension.
     * @param name                  The name to assign to this JMatrix.
     */
    public JMatrix(float[] matrix, int length, int channels, int height, int width, String name) {
        this.matrix = matrix;
        this.length = length;
        this.channels = channels;
        this.height = height;
        this.width = width;
        this.name = name;
    }

    protected float access(int index) {
        // Groundwork for new features
        return matrix[index];
    }

    /**
     * Access the wrapped array.
     */
    public float[] getMatrix() {
        return matrix;
    }

    /**
     * Name this JMatrix
     * @param name          The name to assign.
     */
    public JMatrix setName(String name) {
        this.name = name;
        return this; // For chaining
    }

    /**
     * Access the name of this JMatrix. 
     * @return the name if set. <li> otherwise null.
     */
    public String getName() {
        return name;
    }

    /**
     * The shape of the JMatrix.
     * @returns {length, channels, height, width} in an int[4].
     */
    public int[] shape() {
        return new int[]{length, channels, height, width};
    }

    /**
     * The shape of the JMatrix as a String.
     * @returns {length, channels, height, width} visually organized into a String.
     */
    public String shapeAsString() {
        return "(" + length + "," + channels + "," + height + "," + width + ")";
    }
    /**
     * Print the shape of the JMatrix in the format (length, channels, height, width).
     */
    public void printShape() {
        System.out.println("(" + length + "," + channels + 
            "," + height + "," + width + ")");
    }

    /**
     * Set the wrapped array to a new value. Resize not allowed.
     * @param matrix                            The new array to replace the original. 
     * @exception IllegalArgumentException      if the number of items 
     * in the new array doesn't match the original.
     */
    public void setMatrix(float[] matrix) {
        if (matrix.length != size()) {
            throw new IllegalArgumentException(
                "Sizes must match. Original: " 
                + size() + " New: " + matrix.length
            );
        }
        this.matrix = matrix;
    }

    /**
     * Set the wrapped array to a new value. Resize allowed.
     * @param matrix                            The new array to replace the original. 
     * @param shape                             The four dimensional shape of the new matrix.
     * @exception IllegalArgumentException      if: <p> <ul> <li>  the length of shape is not four. <p> <li>
     *  the reported number of elements is unequal to the length of the matrix. <ul>
     */
    public void setMatrix(float[] matrix, int[] shape) {
        int newSize = length * channels * height * width;
        if (matrix.length != newSize) {
            throw new IllegalArgumentException(
                "Sizes must match. Reported: " 
                + newSize + " Actual: " + matrix.length
            );
        }
        this.matrix = matrix;
        this.length = shape[0];
        this.channels = shape[1];
        this.height = shape[2];
        this.width = shape[3];
    }

    /**
     * The total number of elements.
     */
    public int size() {
        return matrix.length;
    }

    /**
     * The specified batch dimension.
     */
    public int length() {
        return length;
    }
    /**
     * The specified channel dimension.
     */
    public int channels() {
        return channels;
    }
    /**
     * The specified height dimension.
     */
    public int height() {
        return height;
    }
    /**
     * The specified width dimension.
     */
    public int width () {
        return width;
    }

    /**
     * Get an individual element.
     * @param index               The 1D index of the item to get.
     */
    public float get(int index) {
        return access(index);
    }

    /**
     * Get an individual element.
     * @param lengthIndex               The batch index of the item to get.
     * @param channelIndex              The channel index of the item to get.
     * @param heightIndex               The height index of the item to get.
     * @param widthIndex                The width index of the item to get.
     */
    public float get(int lengthIndex, int channelIndex, int heightIndex, int widthIndex) {
        return access(lengthIndex * channels * height * 
            width + channelIndex * height * width + 
            heightIndex * width + widthIndex);
    }
    /**
     * Get a channels * height * width element.
     * @param lengthIndex The index along the batch dimension.
     */
    public double[] getImage(int lengthIndex) {
        int sliceSize = channels * height * width;
        int startIdx = lengthIndex * sliceSize;  
        double[] slice = new double[sliceSize];
        System.arraycopy(matrix, startIdx, slice, 0, sliceSize);
        return slice;
    }


    /**
     * Get a channels * height * width element wrapped in a JMatrix.
     * @param lengthIndex The index along the batch dimension.
     */
     public JMatrix getWrapped(int lengthIndex) {
        int sliceSize = channels * height * width;
        int startIdx = lengthIndex * sliceSize;  
        float[] slice = new float[sliceSize];
        System.arraycopy(matrix, startIdx, slice, 0, sliceSize);
        return new JMatrix(slice, 1, channels, height, width);
    }

    /**
     * Get a height * width element.
     * @param lengthIndex The index along the batch dimension.
     * @param channelIndex The index along the channel dimension.
     */
    public double[] get(int lengthIndex, int channelIndex) {
        int sliceSize = height * width;
        int startIdx = lengthIndex * channels * sliceSize + channelIndex * sliceSize;
        double[] slice = new double[sliceSize];
        System.arraycopy(matrix, startIdx, slice, 0, sliceSize);
        return slice;
    }

    /**
     * Get a height * width element wrapped in a JMatrix.
     * @param lengthIndex The index along the batch dimension.
     * @param channelIndex The index along the channel dimension.
     */
    public JMatrix getWrapped(int lengthIndex, int channelIndex) {
        int sliceSize = height * width;
        int startIdx = lengthIndex * channels * sliceSize + channelIndex * sliceSize;
        float[] slice = new float[sliceSize];
        System.arraycopy(matrix, startIdx, slice, 0, sliceSize);
        return new JMatrix(slice, 1, 1, height, width);
    }

    /**
     * Set a height * width element.
     * @param lengthIndex The index along the batch dimension.
     * @param channelIndex The index along the channel dimension.
     * @param values The values to copy into the specified region.
     * @throws IllegalArgumentException If the number of values is
     * unequal to the height * width of the JMatrix.
     */
    public void set(int lengthIndex, int channelIndex, double[] values) {
        int sliceSize = height * width;
        if (values.length != sliceSize) {
            throw new IllegalArgumentException("Invalid slice size. Expected " + sliceSize + " values.");
        }
        int startIdx = lengthIndex * channels * sliceSize + channelIndex * sliceSize;
        System.arraycopy(values, 0, matrix, startIdx, sliceSize);
    }

    
    /**
     * Alter the dimensional information stored in the JMatrix.
     * @param newLength                 The new batch dimension.
     * @param newChannels               The new channel dimension.
     * @param newHeight                 The new height dimension.
     * @param newWidth                  The new width dimension.
     * @throws IllegalArgumentException If the total size of the reshape 
     * is unequal to that of the original.
     * @return                          A new JMatrix with the changes applied.
     */
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

     /**
     * Alter the dimensional information stored in the JMatrix.
     * @param newLength                 The new shape;
     * @throws IllegalArgumentException If the total size of the reshape 
     * is unequal to that of the original, or if shape.length != 4.
     * @return                          A new JMatrix with the changes applied.
     */
    public JMatrix reshape(int[] shape) {
        if (shape.length != 4) {
            throw new IllegalArgumentException(
                "Invalid shape: length does not equal 4.");
        }
        return reshape(shape[0], shape[1], shape[2], shape[3]);
    }
    /**
     * Set an item with 1D indexing.
     * @param index             The 1D index to alter.
     * @param value             The value to set, cast to a float.            
     */
    public void set(int index, double value) {
        matrix[index] = (float)value;
    }

    /**
     * Set an item with 4D indexing.
     * @param lengthIndex               The batch index of the item to set.
     * @param channelIndex              The channel index of the item to set.
     * @param heightIndex               The height index of the item to set.
     * @param widthIndex                The width index of the item to set.
     * @param value             The value to set, cast to a float.            
     */
    public void set(int lengthIndex, int channelIndex, int widthIndex, int heightIndex, double value) {
        matrix[lengthIndex * channels * height * 
        width + channelIndex * height * width + 
        heightIndex * width + widthIndex] = (float)value;
    }


// Statistics

    /**
     * The max value in the JMatrix.             
     */
    public double max() {
        double max = Double.NEGATIVE_INFINITY;
        for (double d : matrix) {
            max = Math.max(max, d);
        }
        return max;
    }
    /**
     * The max absolute value in the JMatrix.             
     */
    public double absMax() {
        double max = Double.NEGATIVE_INFINITY;
        for (double d : matrix) {
            max = Math.max(max, Math.abs(d));
        }
        return max;
    }
    /**
     * The 1D index of the max value.
     */
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
    /**
     * Finds the index of max values along a given axis.
     * @param axis              The specified axis in the range [0,3] inclusive.
     * @return                  An array containing the indexes of max values along the given axis
     */
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
    /**
     * The mean of all values in the JMatrix.
     */
    public double mean() {
        double mean = 0;

        int size = size();
        for (int i = 0; i < size; i++) {
            mean += access(i);
        
        }
        return mean / size;
    }
    /**
     * The sum of all values in the JMatrix.
     */
    public double sum() {
        double sum = 0;

        int size = size();
        for (int i = 0; i < size; i++) {
            sum += access(i);
        
        }
        return sum;
    }
   
    /**
     * The frobenius norm is calculated as: <p>
     * For every x -> <p>
     * - Raise x ^ 2. <p>
     * - Add it to the sum. <p>
     * Finally, square the sum.
     */
    public float frobeniusNorm() {
        float sum = 0;

        int size = size();
        for (int i = 0; i < size; i++) {
            float pixel = access(i);
            sum += pixel * pixel;
        
        }
        return (float)Math.sqrt(sum);
    }

    /**
     * The l1 norm is calculated as: <p>
     * For every x -> <p>
     * Take the abosolute value and add it to the sum.
     * 
     */
    public float l1Norm() {
        float sum = 0;
        int size = size();
        for (int i = 0; i < size; i++) {
            sum += Math.abs(access(i));
        }
        return sum;
    }

    /**
     * Count the number of items in the JMatrix of a certain value.
     * @param value             The value to count instances of.
     */
    public int count(int value) {
        int count = 0;
        int size = size();
        for (int i = 0; i < size; i++) {
            if (access(i) == value) {
                count++;
            }
        }
        return count;
    }

    /**
     * Scale values in the JMatrix from [0, n] to [0, 1].
     * @return A new JMatrix with the changes applied.
     */
    public JMatrix scaleSigmoid() {
        double max = max();
        return multiply(1.0 / max);
    }

    /**
     * Add Gaussian noise to each item in the JMatrix.
     * @param mean              The mean of the Gaussian noise.
     * @param stdDev            The standard deviation of the Gaussian noise.
     * @return A new JMatrix with the changes applied.
     */
    public JMatrix addGaussianNoise(double mean, double stdDev) {
        JMatrix noisy = this.zerosLike();
        for (int i = 0; i < size(); i++) {
            noisy.set(i, access(i) + (float)(rand.nextGaussian() * stdDev + mean));
        }
        return noisy;
    }

    /**
     * Sum values along rows for 2D use cases.
     * @param scale Whether or not to scale results by 1 / numRows.
     */
    public float[] sum0(boolean scale) {
        int rows = length;
        int cols = channels * height * width;

        float scaleFactor = (scale) ? 1.0f / rows : 1;

        float[] sum = new float[rows];
        IntStream.range(0, rows).parallel().forEach(i -> { 
            for (int j = 0; j < cols; j++) {
                sum[i] = access(i * cols + j);
            }
            sum[i] *= scaleFactor;
        });
        return sum;
    }
    /**
     * Rotate the JMatrix 90 degrees clockwise for 2D use cases.
     * @return A new JMatrix with the changes applied.
     */
    public JMatrix transpose2D() {
        int oldHeight = length;
        int oldWidth = channels * height * width;
        int newHeight = oldWidth;
        int newWidth = oldHeight;

        float[] rotated = new float[size()];

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
    /**
     * Compare the shape of two JMatrixes.
     * @param other             The JMatrix to compare this JMatrix with.
     * @return                  True if the JMatrixes have the same shape.
     */
    public boolean isSameShapeAs(JMatrix other) {
        return length == other.length() && channels == other.channels()
            && height == other.height() && width == other.width();
    }

    /**
     * Returns an exact copy of this JMatrix.
     */
    public JMatrix copy() {
        return new JMatrix(matrix.clone(), length, channels, height, width);
    }
    /**
     * Returns a new empty JMatrix with the same dimensions as this JMatrix.
     */
    public JMatrix zerosLike() {
        return new JMatrix(new float[size()], length, channels, height, width);
    }

    /**
     * Clips all values in the JMatrix to a desired range.
     * @param min               The minimum allowed value, cast to a float.
     * @param max               The maximum allowed value, cast to a float.
     */
    public JMatrix clip(double min, double max) {
        float fMin = (float)min;
        float fMax = (float)max;
        IntStream.range(0, matrix.length).parallel().forEach(i -> {
            matrix[i] = Math.max(fMin, Math.min(fMax, matrix[i]));
        });
        return this; // For chaining
    }

    /**
     * Fills the JMatrix with a certain value.
     * @param fillValue             The value to assign to all items of the JMatrix.
     */
    public JMatrix fill(double fillValue) {
        float valueF = (float)fillValue;
        IntStream.range(0, matrix.length).parallel().forEach(i -> {
            matrix[i] = valueF;
        });
        return this; // For chaining
    }

    /**
     * Perform matrix multiplication with another JMatrix for 2D use cases.
     * @param secondMatrix             The second JMatrix to perform matrix multiplication with.
     * @param scale                    Whether or not to scale values by 1 / rows.
     * @return                         A new JMatrix representing the dot product.
     */
    public JMatrix matmul(JMatrix secondMatrix, boolean scale) {
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

        float scaleFactor = (float)(1.0f / Math.sqrt(k));

        float[] result = new float[m * n];

        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = 0; j < n; j++) {
                float sum = 0;
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
    /**
     * Set every item x in the JMatrix to 1 / x.
     * @return A new JMatrix with the changes applied.
     */
    public JMatrix reciprocal() {
        int size = size();
        float[] result = new float[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = (float)(1.0 / access(i));
        });

        return new JMatrix(result, length, channels, height, width);
    }

    /**
     * Set every item x in the JMatrix to x ^ 1/2.
     * @return A new JMatrix with the changes applied.
     */
    public JMatrix sqrt() {
        int size = size();
        float[] result = new float[size];

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = (float)(Math.sqrt(access(i)));
        });

        return new JMatrix(result, length, channels, height, width);
    }
 
    /**
     * Perform broadcast subtraction with another JMatrix.
     * @param secondMatrix              The JMatrix to subtract from this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match and (1,C,1,1) match are supported.
     * @return                          A new JMatrix representing the difference.
     */
    public JMatrix subtract(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            float[] result = new float[size];
            IntStream.range(0, size).parallel().forEach(i -> {
                result[i] = access(i) - secondMatrix.access(i);
            });
            return new JMatrix(result, length, channels, height, width);
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            float[] result = new float[size];
            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float subtractor = secondMatrix.access(c); // one subtractor per channel
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
            " cannot be broadcast together. Supported: full match or (1,channels,1,1)."
        );
    }

        /**
     * Perform broadcast subtraction with another JMatrix in place.
     * @param secondMatrix              The JMatrix to subtract from this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match and (1,C,1,1) match are supported.
     */
    public JMatrix subtractInPlace(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            IntStream.range(0, size).parallel().forEach(i -> {
                matrix[i] = access(i) - secondMatrix.access(i);
            });
            return this; // For chaining
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float subtractor = secondMatrix.access(c); // one subtractor per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        matrix[offset + i] = access(offset + i) - subtractor;
                    }
                }
            });
            return this; // For chaining
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match or (1,channels,1,1)."
        );
    }


    /**
     * Perform broadcast addition with another JMatrix.
     * @param secondMatrix              The JMatrix to add to this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match and (1,C,1,1) match are supported.
     * @return                          A new JMatrix representing the sum.
     */
     public JMatrix add(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise addition
        if (size == secondMatrix.size()) {
            float[] result = new float[size];
            IntStream.range(0, size).parallel().forEach(i -> {
                result[i] = access(i) + secondMatrix.access(i);
            });
            return new JMatrix(result, length, channels, height, width);
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            float[] result = new float[size];
            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float adder = secondMatrix.access(c); // one adder per channel
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
            " cannot be broadcast together. Supported: full match or (1,channels,1,1)."
        );
    }

    /**
     * Perform broadcast addition with another JMatrix in place.
     * @param secondMatrix              The JMatrix to add to this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match and (1,C,1,1) match are supported.
     */
    public JMatrix addInPlace(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            IntStream.range(0, size).parallel().forEach(i -> {
                matrix[i] = access(i) + secondMatrix.access(i);
            });
            return this; // For chaining
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float adder = secondMatrix.access(c); // one adder per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        matrix[offset + i] = access(offset + i) + adder;
                    }
                }
            });
            return this; // For chaining
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match or (1,channels,1,1)."
        );
    }
    /**
     * Perform broadcast multiplication with another JMatrix.
     * @param secondMatrix The JMatrix to multiply with this JMatrix.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match, (1,C,1,1) match, and (N,C,1,1) match are supported.
     * @return A new JMatrix representing the product.
     */
    public JMatrix multiply(JMatrix secondMatrix) {
        int size = size();
        // Full element-wise multiplication
        if (size == secondMatrix.size()) {
            float[] result = new float[size];
            IntStream.range(0, size).parallel().forEach(i -> {
                result[i] = access(i) * secondMatrix.access(i);
            });
            return new JMatrix(result, length, channels, height, width);
        }
        
        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {
            float[] result = new float[size];
            int channelSize = height * width;
            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float multiplier = secondMatrix.access(c); // one multiplier per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        result[offset + i] = access(offset + i) * multiplier;
                    }
                }
            });
            return new JMatrix(result, length, channels, height, width);
        }
        
        // Sample-channel-wise broadcasting: (N, C, 1, 1) over (N, C, H, W)
        // Each sample has its own set of channel multipliers
        if (secondMatrix.length() == length && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {
            float[] result = new float[size];
            int channelSize = height * width;
            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    // Get multiplier for this specific sample and channel
                    float multiplier = secondMatrix.access(n * channels + c);
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
            " cannot be broadcast together. Supported: full match, (1,C,1,1), or (N,C,1,1)."
        );
    }

    /**
     * Perform broadcast multiplication with another JMatrix in place.
     * @param secondMatrix              The JMatrix to multiply this JMatrix with.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match and (1,C,1,1) match are supported.
     */
    public JMatrix multiplyInPlace(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            IntStream.range(0, size).parallel().forEach(i -> {
                matrix[i] = access(i) * secondMatrix.access(i);
            });
            return this; // For chaining
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float multiplier = secondMatrix.access(c); // one multiplier per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        matrix[offset + i] = access(offset + i) * multiplier;
                    }
                }
            });
            return this; // For chaining
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match or (1,channels,1,1)."
        );
    }

    /**
     * Perform broadcast division with another JMatrix.
     * @param secondMatrix              The JMatrix to divide this JMatrix by.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match and (1,C,1,1) match are supported.
     * @return                          A new JMatrix representing the dividend.
     */
    public JMatrix divide(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            float[] result = new float[size];
            IntStream.range(0, size).parallel().forEach(i -> {
                result[i] = access(i) / secondMatrix.access(i);
            });
            return new JMatrix(result, length, channels, height, width);
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            float[] result = new float[size];
            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float divisor = secondMatrix.access(c); // one divisor per channel
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
            " cannot be broadcast together. Supported: full match or (1,channels,1,1)."
        );
    }

    /**
     * Perform broadcast division with another JMatrix in place.
     * @param secondMatrix              The JMatrix to divide this JMatrix by.
     * @throws IllegalArgumentException If the JMatrixes are incompatible. Full match and (1,C,1,1) match are supported.
     */
    public JMatrix divideInPlace(JMatrix secondMatrix) {
        int size = size();

        // Full element-wise subtraction
        if (size == secondMatrix.size()) {
            IntStream.range(0, size).parallel().forEach(i -> {
                matrix[i] = access(i) / secondMatrix.access(i);
            });
            return this; // For chaining
        }

        // Channel-wise broadcasting: (1, C, 1, 1) over (N, C, H, W)
        if (secondMatrix.length() == 1 && secondMatrix.channels() == channels &&
            secondMatrix.height() == 1 && secondMatrix.width() == 1) {

            int channelSize = height * width;

            IntStream.range(0, length).parallel().forEach(n -> {
                for (int c = 0; c < channels; c++) {
                    float divisor = secondMatrix.access(c); // one divisor per channel
                    int offset = n * channels * channelSize + c * channelSize;
                    for (int i = 0; i < channelSize; i++) {
                        matrix[offset + i] = access(offset + i) / divisor;
                    }
                }
            });
            return this; // For chaining
        }

        // If matrices are not broadcastable
        throw new IllegalArgumentException(
            "Sizes " + size + " and " + secondMatrix.size() +
            " cannot be broadcast together. Supported: full match or (1,channels,1,1)."
        );
    }

    /**
     * Subtract a scalar from this JMatrix.
     * @param scalar              The scalar value to subtract from this JMatrix.
     * @return                    A new JMatrix with the changes applied.
     */
    public JMatrix subtract(double scalar) {
        int size = size();
        float[] result = new float[size];

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = access(i) - fScalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }

    /**
     * Subtract a scalar from this JMatrix in place.
     * @param scalar              The scalar value to subtract from this JMatrix.
     */
    public JMatrix subtractInPlace(double scalar) {
        int size = size();

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            matrix[i] = access(i) - fScalar;
        });

        return this; // For chaining
    }

    /**
     * Add a scalar to this JMatrix.
     * @param scalar              The scalar value to add to this JMatrix.
     * @return                    A new JMatrix with the changes applied.
     */
    public JMatrix add(double scalar) {
        int size = size();
        float[] result = new float[size];

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = access(i) + fScalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }

    /**
     * Add a scalar to this JMatrix in place.
     * @param scalar              The scalar value to add to this JMatrix.
     */
    public JMatrix addInPlace(double scalar) {
        int size = size();

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            matrix[i] = access(i) + fScalar;
        });

        return this; // For chaining
    }
    /**
     * Multiply a scalar with this JMatrix.
     * @param scalar              The scalar value to muliply with this JMatrix.
     * @return                    A new JMatrix with the changes applied.
     */
    public JMatrix multiply(double scalar) {
        int size = size();
        float[] result = new float[size];

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = access(i) * fScalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }

    /**
     * Multiply a scalar with this JMatrix in place.
     * @param scalar              The scalar value to subtract from this JMatrix.
     */
    public JMatrix multiplyInPlace(double scalar) {
        int size = size();

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            matrix[i] = access(i) * fScalar;
        });
        return this; // For chaining
    }

     /**
     * Divide this JMatrix by a scalar.
     * @param scalar              The scalar value to divide this JMatrix by.
     * @return                    A new JMatrix with the changes applied.
     */
    public JMatrix divide(double scalar) {
        int size = size();
        float[] result = new float[size];

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            result[i] = access(i) / fScalar;
        });

        return new JMatrix(result, length, channels, height, width);
    }

    /**
     * Divide this matrix by a scalar in place.
     * @param scalar              The scalar value to divide this JMatrix by.
     */
    public JMatrix divideInPlace(double scalar) {
        int size = size();

        float fScalar = (float)scalar;

        IntStream.range(0, size).parallel().forEach(i -> {
            matrix[i] = access(i) / fScalar;
        });
        
        return this; // For chaining
    }
}
