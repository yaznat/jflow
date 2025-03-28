package JFlow.Layers;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;

import JFlow.JMatrix;
import JFlow.data.Dataloader;
import JFlow.data.Image;
import java.util.List;

public class Sequential {
    private ArrayList<Layer> layers = new ArrayList<Layer>();
    private int numDense;

    public Sequential(){}

    // Add layer to the model
    public void add(Layer layer) {
        if (layer instanceof Dense) {
            numDense++;
        }
        layer.setIDnum(numDense);
        // Link layers
        if (!layers.isEmpty()) {
            layers.getLast().setNextLayer(layer);
            layer.setPreviousLayer(layers.getLast());
        }
        layers.add(layer);
    }
    public void add(Activation activation) {
        layers.getLast().setActivation(activation);
    }
    public void add(Dropout dropout) {
        layers.getLast().setDropout(dropout);
    }


    // public void train(Dataloader loader, int epochs, double learningRate) {
    //     // preload batches
    //     int numBatches = loader.numBatches();
    //     int batchSize = loader.getBatches().get(0).size();
    //     int imageSize = loader.getBatches().get(0).get(0).getHeight()
    //         * loader.getBatches().get(0).get(0).getWidth();
    //     double[][][] xBatches = new double[numBatches][imageSize][batchSize];
    //     int[][] yBatches = new int[numBatches][batchSize];

    //     int count = 0;
    //     for (List<Image> batch : loader.getBatches()) {
    //         double[][] images = new double[batchSize][imageSize];
    //         int[] labels = new int[batchSize];
    //         for (int i = 0; i < batch.size(); i++) {
    //             images[i] = batch.get(i).getFlat();
    //             labels[i] = batch.get(i).getLabel();
    //         }
    //         xBatches[count] = Utility.transpose(images);
    //         yBatches[count++] = labels;
    //     }
    //     // begin training
    //     for (int epoch = 1; epoch <= epochs; epoch++) {
    //         double accuracy = 0;
    //         for (int batch = 0; batch < numBatches; batch++) {
    
    //             layers.getFirst().forward(xBatches[batch], true);
    //             layers.getLast().backward(oneHotEncode(yBatches[batch], 10, true), learningRate * 0.85);


    //             double[][] output = layers.getLast().getOutput();

    //             accuracy += getAccuracy(argmax0(output), yBatches[batch]);
    //         }
    //         if (epoch % 10 == 0) {
    //             System.out.println("\nEpoch: " + epoch + "/" + epochs + 
    //                 " Accuracy: " + (accuracy / loader.numBatches()));
    //         }
    //     }
    // }
    public void train(Dataloader loader, int epochs, double learningRate) {
        // preload batches
        int numBatches = loader.numBatches();
        int batchSize = loader.getBatches().get(0).size();
        int channels = loader.getBatches().get(0).get(0).numChannels();
        int imageHeight = loader.getBatches().get(0).get(0).getHeight();
        int imageWidth = loader.getBatches().get(0).get(0).getWidth();
        double[][] xBatches = new double[numBatches][batchSize * channels * imageHeight * imageWidth];
        int[][] yBatches = new int[numBatches][batchSize];

        int count = 0;
        for (List<Image> batch : loader.getBatches()) {
            double[] images = new double[batchSize * channels * imageHeight * imageWidth];
            int[] labels = new int[batchSize];
            
            for (int i = 0; i < batchSize; i++) {
                double[] image = batch.get(i).getFlat();
                
                int startIdx = i * channels * imageHeight * imageWidth; 
                
                System.arraycopy(image, 0, images, startIdx, image.length);
                
                labels[i] = batch.get(i).getLabel();
            }
            
            xBatches[count] = images;
            yBatches[count++] = labels;
        }

        int numClasses = max(yBatches) + 1;
        // begin training
        for (int epoch = 1; epoch <= epochs; epoch++) {
            double accuracy = 0;
            long startTime = System.nanoTime();
            for (int batch = 0; batch < numBatches; batch++) {
    
                layers.getFirst().forward(new JMatrix(xBatches[batch], 
                    batchSize, channels, imageHeight, imageWidth), true);
                layers.getLast().backward(oneHotEncode(yBatches[batch], numClasses, true), learningRate);


                JMatrix output = layers.getLast().getOutput();

                accuracy += getAccuracy(argmax0(output), yBatches[batch]);

                System.out.print("\rEpoch: " + epoch + "/" + epochs + 
                    " Batch: " + (batch + 1) + "/" + numBatches);
            }
            System.out.println("\nAccuracy: " + (accuracy / loader.numBatches()));
            long endTime = System.nanoTime();
            long duration = endTime - startTime;
            System.out.println("Elapsed time: " + duration * 0.000001 + " miliseconds");
        }
    }

    // Predict classes on 2D flat images
    public int[] predict (double[][] images) {
        int height = images.length;
        int width = images[0].length;
        double[] flattened = new double[height * width];

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                flattened[row * width + col] = images[row][col];
            }
        }
        layers.getFirst().forward(new JMatrix(flattened, height, width, 1, 1), false);
        return argmax0(layers.getLast().getOutput());
    }

    public void setDebug(boolean debug) {
        for (Layer l : layers) {
            l.setDebug(debug);
        }
    }

    // Predict classes on 4D images
    public int[] predict(double[][][][] images) {
        int batchSize = images.length;
        int channels = images[0].length;
        int imageHeight = images[0][0].length;
        int imageWidth = images[0][0][0].length;
        double[] flattened = new double[batchSize * channels * imageHeight * imageWidth];
    
        // Flatten the input images
        for (int i = 0; i < batchSize; i++) {
            int startIdx = i * channels * imageHeight * imageWidth;
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < imageHeight; h++) {
                    for (int w = 0; w < imageWidth; w++) {
                        int flatIndex = startIdx + (c * imageHeight * imageWidth) + (h * imageWidth) + w;
                        flattened[flatIndex] = images[i][c][h][w];
                    }
                }
            }
        }
    
        // Forward pass
        layers.getFirst().forward(new JMatrix(flattened, batchSize, channels, imageHeight, imageWidth), false);
    
        // Get predictions
        return argmax0(layers.getLast().getOutput());
    }
    

    // One hot encode labels
    private JMatrix oneHotEncode(int[] labels, int numClasses,
                                 boolean transpose) throws IllegalArgumentException {
        JMatrix oneHot = new JMatrix(labels.length, numClasses, 1, 1);
        double[] oneHotMatrix = oneHot.getMatrix();
        for (int x = 0; x < labels.length; x++) {
            oneHotMatrix[x * numClasses + labels[x]] = 1.0;
        }
        if (transpose) {
            oneHot = oneHot.transpose2D();
        }
        return oneHot;
    }

    public JMatrix forward(JMatrix images, boolean training) {
        layers.getFirst().forward(images, training);
        return layers.getLast().getOutput();
    }

    public void backward(JMatrix yTrue, double learningRate) {
        layers.getLast().backward(yTrue, learningRate);
    }

    public void backward(int batchSize, double learningRate, int label, int numClasses) {
        int[] labels = new int[batchSize];
        Arrays.fill(labels, label);
        boolean transpose = (layers.getLast() instanceof Dense) ? true : false;
        if (layers.getLast().getActivation() instanceof Softmax) {
            layers.getLast().backward(oneHotEncode(labels, numClasses, true), learningRate);
        }
        else {
            double[] expandedLabels;
            expandedLabels = new double[labels.length];
            for (int i = 0; i < labels.length; i++) {
                expandedLabels[i] = label;
            }
            JMatrix labelMatrix = new JMatrix(expandedLabels, 1, batchSize, 1, 1);
            if (transpose) {
                labelMatrix = labelMatrix.transpose2D();
            }
            layers.getLast().backward(labelMatrix, learningRate);
        }
    }
    
    // Find the max value per column
    private int[] argmax0(JMatrix output) {
        int height = output.length();
        int width = output.channels() * output.height() * output.width();
        double[] arr = output.getMatrix();
        int[] result = new int[width];
    
        for (int col = 0; col < width; col++) {
            double max = Double.NEGATIVE_INFINITY; 
            int index = 0;
    
            for (int row = 0; row < height; row++) {
                int flatIndex = row * width + col; 
                if (arr[flatIndex] > max) {
                    max = arr[flatIndex];
                    index = row;
                }
            }
    
            result[col] = index; 
            // System.out.println(index);
        }
    
        return result;
    }

    public double getAccuracy(int[] predictions, int[] labels) {
        double sum = 0;
        for (int x = 0; x < predictions.length; x++) {
            if (predictions[x] == labels[x]) {
                sum++;
            }
        }
        return sum / predictions.length;
    }

    public JMatrix getLayerOutput(int layerIndex) {
        return layers.get(layerIndex).getOutput();
    }

    public JMatrix getLayerGradient(int layerIndex) {
        return layers.get(0).getGradient();
    }

    // To automate the label reading process
    public int max(int[][] arr) {
        int max = 0;
        for (int[] row : arr) {
            for (int i : row) {
                max = Math.max(max, i);
            }
        }
        return max;
    }
    // Save model weights to text files
    public void saveWeights(String filename) throws IOException {
        BufferedWriter writer = null;
        int numDenseSaved = 0; int numConvSaved = 0;
        for (Layer l : layers) {
            if (l instanceof Dense) {
                numDenseSaved++;
                try{
                    writer = new BufferedWriter(
                            new FileWriter("saved_weights/weights_" + 
                            filename + "/W" + numDenseSaved + "_" + filename + ".txt", false));
                }catch(Exception e){
                    Path path = Paths.get("saved_weights/weights_" + filename);
                    Files.createDirectories(path);
                    writer = new BufferedWriter(
                            new FileWriter("saved_weights/weights_" + 
                            filename + "/W"  + numDenseSaved + "_" + filename + ".txt", false));
                }
                double[] weights = ((Dense)l).getWeights();
                for (double d : weights) {
                    writer.write(d + ",");
                }
                writer.flush();
                writer.close();
                double[] biases = ((Dense)l).getBiases();
                writer = new BufferedWriter(new FileWriter(
                    "saved_weights/weights_" + 
                    filename + "/b" + numDenseSaved + "_" + filename + ".txt", false));
                for (int i = 0; i < biases.length; i++) {
                    writer.write(String.valueOf(biases[i]));
                    writer.newLine();
                }
                writer.flush();
                writer.close();
            } else if (l instanceof Conv2D) {
                numConvSaved++;
                try{
                    writer = new BufferedWriter(
                            new FileWriter("saved_weights/weights_" + 
                            filename + "/Filter" + numConvSaved + "_" + filename + ".txt", false));
                }catch(Exception e){
                    Path path = Paths.get("saved_weights/weights_" + filename);
                    Files.createDirectories(path);
                    writer = new BufferedWriter(
                            new FileWriter("saved_weights/weights_" + 
                            filename + "/Filter"  + numConvSaved + "_" + filename + ".txt", false));
                }
                double[] filters = ((Conv2D)l).getFilters();
                for (int i = 0; i < filters.length; i++) {
                    writer.write(filters[i] + ",");
                }
                writer.flush();
                writer.close();
                double[] biases = ((Conv2D)l).getBiases();
                writer = new BufferedWriter(new FileWriter(
                    "saved_weights/weights_" + 
                    filename + "/FilterBias" + numConvSaved + "_" + filename + ".txt", false));
                for (int i = 0; i < biases.length; i++) {
                    writer.write(String.valueOf(biases[i]));
                    writer.newLine();
                }
                writer.flush();
                writer.close();
            }
        }
        System.out.println("Weights saved to saved_weights/" + filename);
    }
    // Load model weights from text files
    public void loadWeights(String filename) {
        String line;
        int numDenseSaved = 0;
        int numConvSaved = 0;
        for (Layer l : layers) {
            if (l instanceof Dense) {
                numDenseSaved++;
                // load weights
                try (BufferedReader br = new BufferedReader(
                    new FileReader("saved_weights/weights_" + 
                    filename + "/W" + numDenseSaved + "_" + filename + ".txt"))) {
                    double[] weights = ((Dense)l).getWeights();
                    while((line = br.readLine()) != null){
                        double[] loaded = parseDoubleArray(line.split(","));
    
                        System.arraycopy(loaded, 0, weights, 0, loaded.length);
                    }
                } catch (Exception e) {
                    System.out.println(e);
                }
                // load biases
                try (BufferedReader br = new BufferedReader(
                new FileReader("saved_weights/weights_" + 
                filename + "/b" + numDenseSaved + "_" + filename + ".txt"))) {
                    double[] biases = ((Dense)l).getBiases();
                    int x = 0;
                    while((line = br.readLine()) != null){
                        biases[x] = Double.valueOf(line);
                        x++;
                    }
                } catch (Exception e) {
                    System.out.println(e);
                }
            } else if (l instanceof Conv2D) {
                numConvSaved++;
                // load weights
                try (BufferedReader br = new BufferedReader(
                    new FileReader("saved_weights/weights_" + 
                    filename + "/Filter" + numConvSaved + "_" + filename + ".txt"))) {
                    double[] filters = ((Conv2D)l).getFilters();
                    while((line = br.readLine()) != null){
                        double[] flattenedRead = parseDoubleArray(line.split(","));
                        for (int i = 0; i < flattenedRead.length; i++) {
                            filters[i] = flattenedRead[i];
                        }
                    }
                } catch (Exception e) {
                    System.out.println(e);
                }
                // load biases
                try (BufferedReader br = new BufferedReader(
                new FileReader("saved_weights/weights_" + 
                filename + "/FilterBias" + numConvSaved + "_" + filename + ".txt"))) {
                    double[] biases = ((Conv2D)l).getBiases();
                    int x = 0;
                    while((line = br.readLine()) != null){
                        biases[x] = Double.valueOf(line);
                        x++;
                    }
                } catch (Exception e) {
                    System.out.println(e);
                }
            }
        }
    }

    // For reading and writing weights
    public static String arrToString(double[] arr) {
        String result = "";
        for (int x = 0; x < arr.length; x++) {
            result += String.valueOf(arr[x]);
            if (x != arr.length - 1) {
                result += ",";
            }
        }
        return result;
    }
    public static double[] parseDoubleArray(String[] arr) {
        double[] result = new double[arr.length];
        for (int x = 0; x < arr.length; x++) {
            result[x] = Double.parseDouble(arr[x]);
        }
        return result;
    }

    
}
