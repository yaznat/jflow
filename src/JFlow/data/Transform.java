package JFlow.data;
import java.util.ArrayList;
import java.util.function.Function;

public class Transform {
    // Store transforms
    private ArrayList<Function<double[][][], double[][][]>> transforms;
    public Transform() {
        transforms = new ArrayList<Function<double[][][], double[][][]>>();
    }

    protected ArrayList<Function<double[][][], double[][][]>> getTransforms() {
        return transforms;
    }

    // Normalize data to [0, 1]
    public void normalizeSigmoid() {

        transforms.add(
            image -> {
                return DataUtility.multiply(image, 1.0 / 255);

            }
        );
    }

    // Normalize data to [-1, 1]
    public void normalizeTanH() {

        transforms.add(
            image -> {
                return DataUtility.add(DataUtility.multiply(image, 1 / 127.5), -1);
            }
        );
    }

    // Rotate image: 90, 180, or 270 degrees
    public void randomRotation() {
        transforms.add(
            image -> {
                int channels = image.length;
                int numRotations = (int)(Math.random() * 3) + 1;
                double[][][] rotatedImage = copy(image);
                for (int c = 0; c < channels; c++) {
                    for (int r = 0; r < numRotations; r++) {
                        rotatedImage[c] = DataUtility.transpose(rotatedImage[c]);
                    }
                }
                return rotatedImage;
            }
        );
    }

    // Temporary simplified version for increasing size
    public void resize(int height, int width) {
        transforms.add(
            image -> {
                int oldHeight = image[0].length;
                int oldWidth = image[0][0].length;
                int channels = image.length;
                double[][][] resized = new double[channels][height][width];
                for (int c = 0; c < channels; c++) {
                    for (int h = 0; h < oldHeight; h++) {
                        for (int w = 0; w < oldWidth; w++) {
                            resized[c][h][w] = image[c][h][w];
                        }
                    }
                }
                return resized;
            }
        );
    }


    private double[][][] copy(double[][][] arr) {
        int channels = arr.length;
        int height = arr[0].length;
        int width = arr[0][0].length;
        double[][][] copy = new double[channels][height][width];
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    copy[c][h][w] = arr[c][h][w];
                }
            }
        }
        return copy;
    }

    // Resize images by a certain method
    public void resize(int width, int height, String method) {
        // TODO
    }

    // Flattens a 3D array to 1D
    public static double[] flatten3D(double[][][] arr) {
        int d1 = arr.length; int d2 = arr[0].length; int d3 = arr[0][0].length;
        double[] flattened = new double[d1 * d2 * d3];
        int index = 0;
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int k = 0; k < d3; k++) {
                    flattened[index] = arr[i][j][k];
                    index++;
                }
            }
        }
        return flattened;
    }

}
