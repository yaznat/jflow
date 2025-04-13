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
import java.util.HashMap;

import JFlow.JMatrix;
import JFlow.data.Dataloader;
import JFlow.data.Image;
import java.util.List;
import java.util.function.Function;

// The sequential object represents a model
public class Sequential {
    private ArrayList<Layer> layers = new ArrayList<Layer>();
    private ArrayList<Component> components = new ArrayList<Component>();
    private int numClasses = -1;
    private int[] inputShape;

    public Sequential(){}

    // Add layer to the model
    public void add(Layer layer) {
        components.add(layer);
        // Link layers
        if (!layers.isEmpty()) {
            layers.getLast().setNextLayer(layer);
            layer.setPreviousLayer(layers.getLast());
            if (layer instanceof Conv2D) {
                ((Conv2D)layer).build((layers.getLast()).channels());
            }
        } else {
            if (layer instanceof Conv2D) {
                if (inputShape == null) {
                    throw new IllegalArgumentException("Input shape never set");
                }
                ((Conv2D)layer).build(inputShape[0]);
            }
        }
        layers.add(layer);
    }

    public void add(Activation activation) {
        components.add(activation);
        layers.getLast().setActivation(activation);
    }
    public void add(BatchNorm batchNorm) {
        components.add(batchNorm);
        layers.getLast().setBatchNorm(batchNorm);
    }
    public void add(Dropout dropout) {
        components.add(dropout);
        layers.getLast().setDropout(dropout);
    }

    // For when a sample of data might not contain all labels
    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    public void setInputShape(int channels, int height, int width) {
        this.inputShape = new int[]{channels, height, width};
    }


    public void train(Dataloader loader, int epochs, double learningRate) {
        runTraining(loader, epochs, learningRate, null, null);
    }

    public void train(Dataloader loader, int epochs, Function<Integer, Double> scheduler) {
        runTraining(loader, epochs, -1, scheduler, null);
    }

    private void runTraining(Dataloader loader, int epochs, double learningRate, 
        Function<Integer,Double> scheduler, String savePath) {
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

                // Save memory by eliminating duplicate data
                if (loader.isLowMemoryModeOn()) {
                    batch.get(i).unload();
                }
            }
            
            xBatches[count] = images;
            yBatches[count++] = labels;
        }

        int classes = (numClasses == -1) ? max(yBatches) + 1 : numClasses;
        int epochLength = String.valueOf(epochs).length();
        // begin training
        for (int epoch = 1; epoch <= epochs; epoch++) {
            learningRate = (scheduler == null) ? learningRate : scheduler.apply(epoch);
            double accuracy = 0;
            long startTime = System.nanoTime();
            float totalLoss = 0;
            for (int batch = 0; batch < numBatches; batch++) {

    
                layers.getFirst().forward(new JMatrix(xBatches[batch], 
                    batchSize, channels, imageHeight, imageWidth), true);
                layers.getLast().backward(oneHotEncode(yBatches[batch], classes, true), learningRate);


                JMatrix output = layers.getLast().getOutput();

                accuracy += getAccuracy(argmax0(output), yBatches[batch]);

                totalLoss += crossEntropyLoss(output, yBatches[batch]);

                String loss = String.valueOf(totalLoss / (batch + 1));

                // Keep length consistent so reports aren't jumpy
                while (loss.length() < 6) {
                    loss += "0";
                }
                loss = loss.substring(0, 6);

                String epochCount = String.valueOf(epoch);
                while (epochCount.length() < epochLength) {
                    epochCount = "0" + epochCount;
                }

                long batchTime = System.nanoTime();
                long timeSinceStart = batchTime - startTime;
                long timePerBatch = timeSinceStart / (batch + 1);
                long timeRemaining = timePerBatch * (numBatches - (batch + 1));


                String report = "\rEpoch: " + epochCount + "/" + epochs + 
                    " | Batch: " + (batch + 1) + "/" + numBatches + 
                     " | Time Remaining: " + secondsToClock(
                        (int)(timeRemaining * 0.000000001));
                if (scheduler != null) {
                    report += " | lr: " + learningRate;
                }
                report += " | Loss: " + loss;
                System.out.print(report);
            }
            System.out.println("\n      Accuracy: " + (accuracy / loader.numBatches()));
        }
    }

    // Predict classes on 2D flat images
    public int[] predict(double[][] images) {
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
                        try {
                            flattened[flatIndex] = images[i][c][h][w];
                        } catch (ArrayIndexOutOfBoundsException e) {
                        }
                    }
                }
            }
        }
    
        // Forward pass
        layers.getFirst().forward(new JMatrix(flattened, batchSize, channels, imageHeight, imageWidth), false);
    
        // Get predictions
        double[] output = layers.getLast().getOutput().getMatrix();
        if (layers.getLast().getActivation() instanceof Sigmoid) {
            int[] predictions = new int[batchSize];
            for (int i = 0; i < batchSize; i++) {
                predictions[i] = (output[i] >= 0.5) ? 1 : 0;
            }
            return predictions;
        }
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

    public JMatrix backward(JMatrix yTrue, double learningRate) {
        layers.getLast().backward(yTrue, learningRate);
        return layers.getFirst().getGradient();
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
    // Calculate loss per batch
    public static float crossEntropyLoss(JMatrix output, int[] labels) {
        double epsilon = 1e-12;
        int batchSize = labels.length;
        double totalLoss = 0;
        double[] outputMatrix = output.getMatrix();
    
        for (int i = 0; i < batchSize; i++) {
            int label = labels[i];
            int index = label * batchSize + i; // Tranposed
            double predictedProb = outputMatrix[index];
            totalLoss += -Math.log(predictedProb + epsilon);
        }
    
        return (float) (totalLoss / batchSize);
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
        }
    
        return result;
    }

    private double getAccuracy(int[] predictions, int[] labels) {
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
    // Save model weights to .txt files
    public void saveWeights(String path) {
        // Keep track of how many layers of each type have been saved
        HashMap<String, Integer> saved = new HashMap<>();

        for (Component c : components) {
            // Check if the layer needs saving
            if (c.numTrainableParameters() != 0) {
                String layerName = c.getName();
                if (saved.containsKey(layerName)) {
                    // Tick up the counter for the layer
                    saved.put(layerName, saved.get(layerName) + 1);
                } else {
                    // Add the layer to the hashmap
                    saved.put(layerName, 1);
                }
                // Write the weights to a .txt file
                HashMap<String, JMatrix> weights = c.getWeights();
                weights.forEach((key, value) -> {
                    BufferedWriter writer = null;
                    try {
                        try{
                            writer = new BufferedWriter(
                                    new FileWriter("saved_weights/weights_" + 
                                    path + "/" + key + "_" + saved.get(layerName) + 
                                     ".txt", false));
                        }catch(Exception e1){
                            Path dir = Paths.get("saved_weights/weights_" + path);
                            Files.createDirectories(dir);
                            writer = new BufferedWriter(
                                new FileWriter("saved_weights/weights_" + 
                                path + "/" + key + "_" + saved.get(layerName) + 
                                 ".txt", false));
                        }
                        double[] matrix = value.getMatrix();
                        for (int i = 0; i < matrix.length; i++) {
                            writer.write(String.valueOf(matrix[i]) + ",");
                        }
                        writer.flush();
                        writer.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                });

            }
        }
        System.out.println("Weights saved to saved_weights/" + path);
    }
    // Load model weights from .txt files
    public void loadWeights(String path) {
        // Keep track of how many layers of each type have been loaded
        HashMap<String, Integer> loaded = new HashMap<>();

        for (Component c : components) {
            // Check if the layer needs loading
            if (c.numTrainableParameters() != 0) {
                String layerName = c.getName();
                if (loaded.containsKey(layerName)) {
                    // Tick up the counter for the layer
                    loaded.put(layerName, loaded.get(layerName) + 1);
                } else {
                    // Add the layer to the hashmap
                    loaded.put(layerName, 1);
                }
                // Load the layer weights from a .txt file
                HashMap<String, JMatrix> weights = c.getWeights();
                weights.forEach((key, value) -> {
                    try (BufferedReader br = new BufferedReader(
                        new FileReader("saved_weights/weights_" + path + "/" + 
                                        key + "_" + loaded.get(layerName) + ".txt"))) {
                        double[] matrix = value.getMatrix();
                        String line; 
                        String[] split = new String[0];
                        while((line = br.readLine()) != null){
                            split = line.split(",");
                        }
                        int x = 0;
                        for (String s : split) {
                            matrix[x++] = Double.parseDouble(s);
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                });

            }
        }
    }

    // Load and build
    public void load(String path) {
        components.clear();
        layers.clear();
    }

    // For time reports while training
    private String secondsToClock(int totalSeconds) {
        int hours = 0; int minutes = 0;
        // hours
        if (totalSeconds > 3600) {
            int hoursDiv = totalSeconds / 3600;
            totalSeconds -= 3600 * hoursDiv;
            hours += hoursDiv;
        } else if (totalSeconds == 3600) {
            hours++;
            totalSeconds = 0;
        }
        // minutes
        if (totalSeconds > 60) {
            int minutesDiv = totalSeconds / 60;
            totalSeconds -= 60 * minutesDiv;
            minutes += minutesDiv;
        } else if (totalSeconds == 60) {
            minutes++;
            totalSeconds = 0;
        }
        if (hours != 0) {
            return hours + ":" + ((minutes < 10) ? "0" + minutes : "" + minutes);
        }
        return ((minutes < 10) ? "0" + minutes : "" + minutes) + ":" + 
            ((totalSeconds < 10) ? "0" + totalSeconds : "" + totalSeconds);
    }

    // print a model summary in the terminal
    public void summary() {
        // Run an empty JMatrix through the model
        if (inputShape != null) {
            JMatrix empty = new JMatrix(1, inputShape[0], inputShape[1], inputShape[2]);
                layers.getFirst().forward(empty, false);
        }
        // Sum trainable paramters
        int trainableParameters = 0;
        for (Component c : components) {
            trainableParameters += c.numTrainableParameters();
        }
        // Declare the size of each column
        int spacesType = 40;
        int spacesShape = 30;
        int spacesParam = 20;
        // Display the layer names and types
        String[] layerTypes = new String[components.size() + 1];
        layerTypes[0] = "Layer (type)";
        HashMap<String, Integer> layerCounts = new HashMap<>();
        for (int i = 0; i < components.size(); i++) {
            String type = components.get(i).getClass().getSimpleName();
            if(layerCounts.containsKey(type)) {
                layerCounts.put(type, layerCounts.get(type) + 1);
            } else {
                layerCounts.put(type, 1);
            }
            String name = components.get(i).getName() + "_" + layerCounts.get(type);
            layerTypes[i + 1] = name + " (" + type + ")";
        }

        String[] shapes = new String[components.size() + 1];
        shapes[0] = "Output Shape";
        for (int i = 0; i < components.size(); i++) {
            int[] outputShape = components.get(i).getOutputShape();
            String shapeAsString = "(None,";
            for (int j = 1; j < outputShape.length; j++) {
                shapeAsString += outputShape[j];
                if (j != outputShape.length - 1) {
                    shapeAsString += ",";
                }
            }
            shapeAsString += ")";
            shapes[i + 1] = shapeAsString;
        }
        String[] params = new String[components.size() + 1];
        params[0] = "Param #";
        for (int i = 0; i < components.size(); i++) {
            params[i + 1] = String.valueOf(components.get(i).numTrainableParameters());
        }
        System.out.println("\033[0;1mModel Summary\033[0m");
        for (int i = 0; i < spacesType + spacesShape + spacesParam + 2; i++) {
            System.out.print("-");
        }
        System.out.print("\n");
        for (int line = 0; line < layerTypes.length; line++) {
            int numSpaces = spacesType - layerTypes[line].length();
            System.out.print(layerTypes[line]);
            for (int i = 0; i < numSpaces; i++) {
                System.out.print(" ");
            }

            numSpaces = spacesShape - shapes[line].length() - 1;
            System.out.print("| ");
            System.out.print(shapes[line]);
            for (int i = 0; i < numSpaces; i++) {
                System.out.print(" ");
            }

            numSpaces = spacesParam - params[line].length() - 1;
            System.out.print("| ");;
            for (int i = 0; i < numSpaces; i++) {
                System.out.print(" ");
            }
            System.out.print(params[line]);
            System.out.print(" \n");

            for (int i = 0; i < spacesType + spacesShape + spacesParam + 2; i++) {
                System.out.print("-");
            }
            System.out.print("\n");
        }
        // Add commas to the number of parameters for readability
        String parameters = String.valueOf(trainableParameters);
        int numCommas = (parameters.length() - 1) / 3;
            int length = parameters.length();
            for (int j = 0; j < numCommas; j++) {
                int index = 1 + j * 4 + ((length % 3 == 0) ? 2 : length % 3 - 1);
                parameters = parameters.substring(0, index) + "," + parameters.substring(index);
            }
        System.out.println("Total params: " + parameters);
        System.out.println("Trainable params: " + parameters);
        System.out.println("Non-trainable params: " + 0);
    }
}
