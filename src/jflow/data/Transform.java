package jflow.data;
import java.util.ArrayList;
import java.util.function.Function;

public class Transform {
    // Store transforms
    private ArrayList<Function<float[][][], float[][][]>> transforms;
    // Keep track of image bounds for clipping
    private int[] normBounds = new int[2];
    /**
     * Initializes an empty Transform.
     */
    public Transform() {
        transforms = new ArrayList<>();
    }

    protected ArrayList<Function<float[][][], float[][][]>> getTransforms() {
        return transforms;
    }

    // Provide information about 
    protected int[] normBounds() {
        return normBounds;
    }

    /**
     * Normalize image data to [0,1].
     */
    public Transform normalizeSigmoid() {
        normBounds[0] = 0;
        normBounds[1] = 1;
        transforms.add(
            image -> {
                return DataUtility.multiply(image, 1.0 / 255);

            }
        );
        return this;
    }

   /**
     * Normalize image data to [-1,1].
     */
    public Transform normalizeTanh() {
        normBounds[0] = -1;
        normBounds[1] = 1;
        transforms.add(
            image -> {
                return DataUtility.add(DataUtility.multiply(image, 1 / 127.5), -1);
            }
        );
        return this;
    }

    /**
     * Polarize grayscale pixel values to 0 (black) and 255 (white) only.
     */
    public Transform grayscaleFullContrast() {
        transforms.add(
            image -> {
                int channels = image.length;
                int height = image[0].length;
                int width = image[0][0].length;
        
                float[][][] contrasted = new float[channels][height][width];

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
        return this;
    }

    /**
     * Invert RGB values (0 -> 255...)
     */
    public Transform invert() {
        transforms.add(
            image -> {
                int channels = image.length;
                int height = image[0].length;
                int width = image[0][0].length;
        
                float[][][] inverted = new float[channels][height][width];
        
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
        return this;
    }

     /**
     * Rotate images by either 90, 180, or 270 degrees.
     */
    public Transform randomRotation() {
        System.out.println(normBounds[0]);
        transforms.add(
            image -> {
                int channels = image.length;
                int numRotations = (int)(Math.random() * 3) + 1;
                float[][][] rotatedImage = copy(image);
                for (int c = 0; c < channels; c++) {
                    for (int r = 0; r < numRotations; r++) {
                        rotatedImage[c] = DataUtility.transpose(rotatedImage[c]);
                    }
                }
                return rotatedImage;
            }
        );
        return this;
    }
    /**
     * 50% chance to flip the image horizontally.
     */
    public Transform randomFlip() {
        transforms.add(
            image -> {
                if (Math.random() > 0.5) {
                    int channels = image.length;
                    int height = image[0].length;
                    int width = image[0][0].length;
                    float[][][] flipped = copy(image);
                    for (int c = 0; c < channels; c++) {
                        for (int i = 0; i < height; i++) {
                            for (int j = 0; j < width; j++) {
                                flipped[c][i][j] = image[c][i][width - 1 - j];
                            }
                        }
                    }
                    return flipped;
                } else {
                    return image;
                }
            }
        );
        return this;
    }
    /**
     * Adds a random brightness value to images.
     */
    public Transform randomBrightness() {
        transforms.add(
            image -> {
                // Random value from -0.2 to 0.2
                double brightness = Math.random() / 2.5 - 0.2;
                return DataUtility.clip(
                    DataUtility.add(image, brightness), 
                    normBounds[0], normBounds[1]);
            }
        );
        return this;
    }

    /**
     * Resize with nearest-neighbor interpolation.
     */
    public Transform resize(int height, int width) {
        transforms.add(image -> {
            int channels = image.length;
            int oldHeight = image[0].length;
            int oldWidth = image[0][0].length;
            
            float[][][] resized = new float[channels][height][width];
    
            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        int srcY = (int)((i / (float)height) * oldHeight);
                        int srcX = (int)((j / (float)width) * oldWidth);
                        srcY = Math.min(srcY, oldHeight - 1);
                        srcX = Math.min(srcX, oldWidth - 1);
                        
                        resized[c][i][j] = image[c][srcY][srcX];
                    }
                }
            }
            
            return resized;
        });
        return this;
    }
    private float[][][] copy(float[][][] arr) {
        int channels = arr.length;
        int height = arr[0].length;
        int width = arr[0][0].length;
        float[][][] copy = new float[channels][height][width];
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    copy[c][h][w] = arr[c][h][w];
                }
            }
        }
        return copy;
    }
}
