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

    // Polarize grayscale pixel values to black (0) and white (255) only 
    public void grayscaleFullContrast() {
        transforms.add(
            image -> {
                int channels = image.length;
                int height = image[0].length;
                int width = image[0][0].length;
        
                double[][][] contrasted = new double[channels][height][width];

                for (int c = 0; c < channels; c++) {
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            contrasted[c][i][j] = (image[c][i][j] > 0) ? 255 : 0;
                        }
                    }
                }
        
                return contrasted;
            }
        );
    }

    // Invert grayscale values (black -> white, etc.)
    public void grayscaleInvert() {
        transforms.add(
            image -> {
                int channels = image.length;
                int height = image[0].length;
                int width = image[0][0].length;
        
                double[][][] inverted = new double[channels][height][width];
        
                for (int c = 0; c < channels; c++) {
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            inverted[c][i][j] = 255 - image[c][i][j];
                        }
                    }
                }
        
                return inverted;
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
    // 50% chance to flip the image horizontally
    public void randomFlip() {
        transforms.add(
            image -> {
                if (Math.random() > 0.5) {
                    int channels = image.length;
                    double[][][] rotatedImage = copy(image);
                    for (int c = 0; c < channels; c++) {
                        for (int r = 0; r < 2; r++) {
                            rotatedImage[c] = DataUtility.transpose(rotatedImage[c]);
                        }
                    }
                    return rotatedImage;
                } else {
                    return image;
                }
            }
        );
    }
    // Add random brightness to the image and clip to range
    public void randomBrightness() {
        transforms.add(
            image -> {
                double low = DataUtility.min(image);
                double high = DataUtility.max(image);
                // Random value from -0.2 to 0.2
                double brightness = Math.random() / 2.5 - 0.2;
                return DataUtility.clip(DataUtility.add(image, brightness), low, high);
            }
        );
    }

    // Resize with nearest-neighbor interpolation
    public void resize(int height, int width) {
        transforms.add(image -> {
            int channels = image.length;
            int oldHeight = image[0].length;
            int oldWidth = image[0][0].length;
            
            double[][][] resized = new double[channels][height][width];
    
            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        int srcY = (int) Math.round(((double) i / height) * oldHeight);
                        int srcX = (int) Math.round(((double) j / width) * oldWidth);
                        srcY = Math.min(srcY, oldHeight - 1);
                        srcX = Math.min(srcX, oldWidth - 1);
                        
                        resized[c][i][j] = image[c][srcY][srcX];
                    }
                }
            }
            
            return resized;
        });
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
}
