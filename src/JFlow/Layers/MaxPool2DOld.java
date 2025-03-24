// package JFlow.Layers;

// import java.util.stream.IntStream;

// import JFlow.Utility;

// public class MaxPool2D extends Layer {
//     private int poolSize, stride;
//     private double[][][][] output, gradient, lastInput;

//     protected MaxPool2D (int poolSize, int stride) {
//         super (0, "max_pool_2d");
//         this.poolSize = poolSize;
//         this.stride = stride;
//     }

//     @Override
//     public void forward(double[][][][] input, boolean training) {
//         lastInput = input;
//         int numImages = input.length;
//         int channels = input[0].length;
//         int imageHeight = (input[0][0].length - poolSize) / stride + 1;
//         int imageWidth = (input[0][0][0].length - poolSize) / stride + 1;

//         output = new double[numImages][channels][imageHeight][imageWidth];

//         // for (int i = 0; i < numImages; i++) {
//         //     for (int c = 0; c < channels; c++) {
//         //         output[i][c] = maxPool2D(input[i][c], poolSize, stride);
//         //     }
//         // }
//         IntStream.range(0, numImages).parallel().forEach(i -> {
//             for (int c = 0; c < channels; c++) {
//                 output[i][c] = maxPool2D(input[i][c], poolSize, stride);
//             }
//         });

//         if (getNextLayer() != null) {
//             getNextLayer().forward(output, training);
//         }
//     }


//     @Override
//     public void forward(double[][] input, boolean training) {
//         // TODO Auto-generated method stub
//         throw new UnsupportedOperationException("Unimplemented method 'forward'");
//     }

//     public void backward(double[][][][] input, double learningRate) {
//     int numImages = lastInput.length;
//     int channels = lastInput[0].length;
//     int imageHeight = lastInput[0][0].length; 
//     int imageWidth = lastInput[0][0][0].length; 

//     gradient = new double[numImages][channels][imageHeight][imageWidth];

//         // Find max locations to pass the gradients through
//         IntStream.range(0, numImages).parallel().forEach(i -> {
//         for (int c = 0; c < channels; c++) {
//             for (int sX = 0; sX < imageHeight / stride; sX++) {
//                 for (int sY = 0; sY < imageWidth / stride; sY++) {
//                     // Find max value per pool
//                     double max = Double.NEGATIVE_INFINITY;
//                     int indexX = 0, indexY = 0;

//                     for (int poolX = 0; poolX < poolSize; poolX++) {
//                         for (int poolY = 0; poolY < poolSize; poolY++) {
//                             int x = sX * stride + poolX;
//                             int y = sY * stride + poolY;

//                             if (x < imageHeight && y < imageWidth && lastInput[i][c][x][y] > max) {
//                                 max = lastInput[i][c][x][y];
//                                 indexX = poolX;
//                                 indexY = poolY;
//                             }
//                         }
//                     }

//                     // Pass back gradient at index of max value
//                     int targetX = sX * stride + indexX;
//                     int targetY = sY * stride + indexY;

//                     gradient[i][c][targetX][targetY] = input[i][c][sX][sY];
//                 }
//             }
//         }
//     });
//     if (getPreviousLayer() != null) {
//         getPreviousLayer().backward(gradient, learningRate);
//     }
// }

//     @Override
//     public void backward(double[][] input, double learningRate) {
//         int numImages = output.length;
//         int channels = output[0].length;
//         int height = output[0][0].length;
//         int width = output[0][0][0].length;

//         double[][] transposed = Utility.transpose(input);

//         double[][][][] reshaped = new double[numImages][channels][height][width];

//         for (int i = 0; i < numImages; i++) {
//             for (int c = 0; c < channels; c++) {
//                 for (int h = 0; h < height; h++) {
//                     for (int w = 0; w < width; w++) {
//                         int flatIndex = c * (height * width) + h * width + w;
//                         reshaped[i][c][h][w] = transposed[i][flatIndex]; 
//                     }
//                 }
//             }
//         }
//         backward(reshaped, learningRate);
//     }

//     @Override
//     public double[][] getOutput() {
//         // TODO Auto-generated method stub
//         throw new UnsupportedOperationException("Unimplemented method 'getOutput'");
//     }


//     public static double[][] maxPool2D(double[][] image, int poolSize, int stride) {
//         // Initialize resulting image
//         int outputSize = (image.length - poolSize) / stride + 1;
//         double[][] result = new double[outputSize][outputSize];

//         // Perform pooling
//         for (int i = 0; i < outputSize; i++) {
//             for (int j = 0; j < outputSize; j++) {
//                 // Find max value per pool
//                 double maxValue = Double.NEGATIVE_INFINITY;

//                 for (int a = 0; a < stride; a++) {
//                     for (int b = 0; b < stride; b++) {
//                         double pixel = image[i * stride + a][j * stride + b];
//                         if (pixel > maxValue) {
//                             maxValue = pixel;
//                         }
//                     }
//                 }
//                 result[i][j] = maxValue;
//             }
//         }
//         return result;
//     }
    
// }
