package jflow.model;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.function.Function;

import jflow.data.*;
import jflow.layers.internal.Dense;
import jflow.layers.internal.Sigmoid;
import jflow.layers.templates.TrainableLayer;
import jflow.utils.Metrics;



// The sequential object represents a model
public class Sequential{
    private ArrayList<jflow.layers.internal.Layer> layers = new ArrayList<>();
    private int numClasses = -1;
    private String name = null;
    private boolean debugMode;
    private Optimizer optimizer;
    private int[] inputShape;
    private HashMap<String, JMatrix[]> layerGradients = new HashMap<>();
    private HashMap<String, Integer> layerCounts = new HashMap<>();

    /**
     * Initializes an empty Sequential model.
     */
    public Sequential(){}

    /**
     * Initializes an empty Sequential model.
     */
    public Sequential(String name){
        this.name = name;
    }
    /**
     * Add a layer to the model.
     * @param layer A JFlow Layer.
     */
    public Sequential add(Layer layer) {
        jflow.layers.internal.Layer internal = layer.getInternal();
        String name = internal.getName();
        if (layerCounts.containsKey(name)) {
            // Tick up the counter for the layer type
            layerCounts.put(name, layerCounts.get(name) + 1);
        } else {
            // Add the layer type to the hashmap
            layerCounts.put(name, 1);
        }
        // Link layers
        if (layers.isEmpty()) {
            if (inputShape != null) {
                internal.setInputShape(inputShape);
            }
        } else {
            layers.getLast().setNextLayer(internal);
            internal.setPreviousLayer(layers.getLast());
        }
        internal.build();
        layers.add(internal);
        return this;
    }

    /**
     * Set the input shape of the model. <p>
     * Alternative option to declaring input shape in the first layer.
     * @param shape                         An InputShape object.
     */
    public Sequential setInputShape(InputShape shape) {
        this.inputShape = shape.getShape();
        return this;
    }

    /**
     * Prepare the model for training.
     * @param Optimizer The desired optimizer.
     */
    public void compile(Optimizer optimizer) {
        setOptimizer(optimizer);
    }

    // Initialize each trainable layer in the optimizer
    private void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
        for (jflow.layers.internal.Layer l : layers) {
            if (l instanceof TrainableLayer) {
                TrainableLayer trainable = (TrainableLayer)l;
                optimizer.init(trainable);
                /*
                 * Store references to internal gradients.
                 * References always remain valid.
                 */ 
                layerGradients.put(trainable.getName(), 
                    trainable.getParameterGradients());
            }
        }
    }


     /**
     * Retrieve the parameter gradients from the model. <p>
     * For custom train steps, use: optimizer.apply(model.trainableVariables())
     */
    public HashMap<String, JMatrix[]> trainableVariables() {
        return layerGradients;
    }

    private int countNumClasses(Dataloader loader) {
        int numImages = loader.size();
        ArrayList<Integer> uniqueLabels = new ArrayList<Integer>();
        for (int i = 0; i < numImages; i++) {
            int label = loader.get(i).getLabel();
            if (!uniqueLabels.contains(label)) {
                uniqueLabels.add(label);
            }
        }
        return uniqueLabels.size();
    }

    /**
     * Use if current train data does not contain images of all classification labels.
     * @param numClasses Set the number of classification labels.
     */
    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    /**
     * Train the model.
     * @param loader                A Dataloader containing train images.
     * @param epochs                The number of epochs to train.
     */
    public void train(Dataloader loader, int epochs) {
        runTraining(loader, epochs, null, null);
    }

    // /**
    //  * Train the model with a learning rate scheduler.
    //  * @param loader                A Dataloader containing train images.
    //  * @param epochs                The number of epochs to train.
    //  * @param scheduler             A function that returns a Double, learningRate, from an Integer, epoch.
    //  */
    // public void train(Dataloader loader, int epochs, Function<Integer, Double> scheduler) {
    //     runTraining(loader, epochs, scheduler, null);
    // }

    private void runTraining(Dataloader loader, int epochs,
        Function<Integer,Double> scheduler, String savePath) {
        // Ensure there is an optimizer (currently necessary)
        if (optimizer == null) {
            setOptimizer(new Adam(0.001));
        }
        
        // Visually separate training to reduce terminal clutter
        System.out.println("");
        // preload batches
        int numBatches = loader.numBatches();
        int batchSize = loader.getBatch(0).size();

        int classes = (numClasses == -1) ? countNumClasses(loader) : numClasses;
        // begin training
        for (int epoch = 1; epoch <= epochs; epoch++) {
            double accuracy = 0;
            long startTime = System.nanoTime();
            double totalLoss = 0;
            for (int batch = 0; batch < numBatches; batch++) {
                JMatrix xBatch = null;
                int[] yBatch = null;

                xBatch = loader.getBatchFlat(batch);
                yBatch = loader.getBatchLabels(batch);

                forward(xBatch, true);

                JMatrix yTrue;
                if (layers.getLast() instanceof Sigmoid) {
                    float[] yBatchf = new float[batchSize];
                    for (int i = 0; i < batchSize; i++) {
                        yBatchf[i] = (float)yBatch[i];
                    }
                    yTrue = new JMatrix(yBatchf, batchSize, 1, 1, 1);
                    
                } else {
                    yTrue = oneHotEncode(yBatch, classes, true);
                }

                backward(yTrue);

                // Apply updates
                optimizer.apply(layerGradients);

                JMatrix output = layers.getLast().getOutput();

                int[] predictions;

                if (layers.getLast() instanceof Sigmoid) {
                    predictions = new int[batchSize];
                    for (int i = 0; i < batchSize; i++) {
                        predictions[i] = (output.get(i) >= 0.5) ? 1 : 0;
                    }
                } else {
                    predictions = argmax0(output);
                }
                accuracy += Metrics.getAccuracy(predictions, yBatch);

                totalLoss += crossEntropyLoss(output, yBatch);

                long batchTime = System.nanoTime();
                long timeSinceStart = batchTime - startTime;

                HashMap<String, Double> lossReport = new HashMap<>();
                lossReport.put("Loss", totalLoss / (batch + 1));

                if (!debugMode) {
                    if (scheduler == null) {
                        Metrics.printTrainingCallback(epoch, epochs, batch + 1, numBatches,
                            timeSinceStart, lossReport);
                    } else {
                        Metrics.printTrainingCallback(epoch, epochs, batch + 1, numBatches,
                            timeSinceStart, lossReport, scheduler.apply(epoch));
                    }
                }
            }
            System.out.println("\n      Accuracy: " + (accuracy / loader.numBatches()));
        }
    }

    /**
     * Predict class labels on 2D flattened images
     * @param images                Images in the shape (N, flattenedSize).
     * @return                      Predicted class labels from softmax.
     */
    public int[] predict(float[][] images) {
        int height = images.length;
        int width = images[0].length;
        float[] flattened = new float[height * width];

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                flattened[row * width + col] = images[row][col];
            }
        }

        // Forward pass
        JMatrix output = forward(new JMatrix(flattened, height, width, 1, 1), false);

        // Get predictions
        int batchSize = height;
        if (layers.getLast() instanceof Sigmoid) {
            int[] predictions = new int[batchSize];
            for (int i = 0; i < batchSize; i++) {
                predictions[i] = (output.get(i) >= 0.5) ? 1 : 0;
            }
            return predictions;
        }
        return argmax0(output);
    }
    /**
     * Predict class labels on 4D images.
     * @param images                Images in the shape (N, channels, height, width).
     * @return                      Predicted class labels from softmax.
     */
    public int[] predict(float[][][][] images) {
        int batchSize = images.length;
        int channels = images[0].length;
        int imageHeight = images[0][0].length;
        int imageWidth = images[0][0][0].length;
        float[] flattened = new float[batchSize * channels * imageHeight * imageWidth];
    
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
        JMatrix output = forward(new JMatrix(flattened, batchSize, channels, imageHeight, imageWidth), false);

    
        // Get predictions
        if (layers.getLast() instanceof Sigmoid) {
            int[] predictions = new int[batchSize];
            for (int i = 0; i < batchSize; i++) {
                predictions[i] = (output.get(i) >= 0.5) ? 1 : 0;
            }
            return predictions;
        }
        return argmax0(layers.getLast().getOutput());
    }

    /**
     * When enabled, prints statistical data from each layer.
     * @param enabled               Set debug mode to on or off.
     */
    public void setDebugMode(boolean enabled) {
        debugMode = enabled;
    }
    
    // One hot encode labels
    private JMatrix oneHotEncode(int[] labels, int numClasses,
                                 boolean transpose) throws IllegalArgumentException {
        JMatrix oneHot = new JMatrix(labels.length, numClasses, 1, 1);
        float[] oneHotMatrix = oneHot.getMatrix();
        for (int x = 0; x < labels.length; x++) {
            oneHotMatrix[x * numClasses + labels[x]] = 1.0f;
        }
        if (transpose) {
            oneHot = oneHot.transpose2D();
        }
        return oneHot;
    }

    /**
     * Perform forward propagation.
     * @param images               Image data wrapped in a JMatrix.
     * @param training             Indicate whether for training or inference.
     * @return                     Returns the forward output of the last layer of the model.
     */
    public JMatrix forward(JMatrix images, boolean training) {
        JMatrix output = layers.get(0).forward(images, training);;
        for (int i = 1; i < layers.size(); i++) {
            output = layers.get(i).forward(output, training);
        }
        return output;
    }

    /**
     * Perform backward propagation.
     * @param images               yTrue data wrapped in a JMatrix.
     * @param learningRate         The desired learning rate for updating parameters.
     * @return                     Returns the gradient, dX, of the first layer of the model.
     */
    public JMatrix backward(JMatrix yTrue) {
        JMatrix gradient = layers.getLast().backward(yTrue);
        for (int i = layers.size() - 2; i >= 0; i--) {
            gradient = layers.get(i).backward(gradient);
            if (debugMode) {
                layers.get(i).printDebug();
            }
        }
        return gradient;
    }

    // Calculate loss per batch
    private double crossEntropyLoss(JMatrix output, int[] labels) {
        double epsilon = 1e-12;
        int batchSize = labels.length;
        double totalLoss = 0;

        if (layers.getLast() instanceof Sigmoid) {
            // b.c.e. for sigmoid activation
            for (int i = 0; i < batchSize; i++) {
                int label = labels[i];
                double predictedProb = output.get(i);
                
                // b.c.e.
                totalLoss += -label * Math.log(predictedProb + epsilon) - 
                            (1 - label) * Math.log(1 - predictedProb + epsilon);
            }
        } else {
            for (int i = 0; i < batchSize; i++) {
                int label = labels[i];
                int index = label * batchSize + i; // Tranposed
                double predictedProb = output.get(index);
                totalLoss += -Math.log(predictedProb + epsilon);
            }
        }
    
        return totalLoss / batchSize;
    }
    
    // Find the max value per column
    private int[] argmax0(JMatrix output) {
        int height = output.length();
        int width = output.channels() * output.height() * output.width();
        float[] arr = output.getMatrix();
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

    /**
     * Get the forward output of a layer in the model.
     * @param layerIndex               The index of the desired layer.
     */
    public JMatrix getLayerOutput(int layerIndex) {
        return layers.get(layerIndex).getOutput();
    }

    /**
     * Get the forward output of the last layer in the model.
     */
    public JMatrix getLastLayerOutput() {
        return layers.getLast().getOutput();
    }

    /**
     * Get the gradient, dX, of a layer in the model.
     * @param layerIndex               The index of the desired layer.
     */
    public JMatrix getLayerGradient(int layerIndex) {
        return layers.get(0).getGradient();
    }


    /**
     * Save weights to .txt files in a directory.
     * @param path               The name of the directory to store files in.
     */
    public void saveWeights(String path) {
        for (jflow.layers.internal.Layer l : layers) {
            // Check if the layer needs saving
            if (l instanceof TrainableLayer) {
                TrainableLayer trainable = (TrainableLayer)l;
    
                JMatrix[] weights = trainable.getWeights();
                for (JMatrix weight : weights) {
                    BufferedWriter writer = null;
                    try {
                        try{
                            writer = new BufferedWriter(
                                    new FileWriter(path + "/" + trainable.getName() + 
                                    "_" + weight.getName() + ".txt", false));
                        }catch(Exception e1){
                            Path dir = Paths.get(path);
                            Files.createDirectories(dir);
                            writer = new BufferedWriter(
                                new FileWriter( path + "/" + trainable.getName() 
                                + "_" + weight.getName() + ".txt", false));
                        }
                        for (int i = 0; i < weight.size(); i++) {
                            writer.write(String.valueOf(weight.get(i)) + ",");
                        }
                        writer.flush();
                        writer.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

            }
        }
        System.out.println("Weights saved to " + path);
    }
    /**
     * Load weights from .txt files in a directory.
     * @param path               The location of the directory to load files from.
     */
    public void loadWeights(String path) {
        for (jflow.layers.internal.Layer l : layers) {
            // Check if the layer needs loading
            if (l instanceof TrainableLayer) {
                TrainableLayer trainable = (TrainableLayer)l;

                // Load the layer weights from a .txt file
                JMatrix[] weights = trainable.getWeights();
                for (JMatrix weight : weights) {
                    try (BufferedReader br = new BufferedReader(
                        new FileReader(path + "/" + 
                            trainable.getName() + "_" + 
                            weight.getName() + ".txt"))) {
                        String line; 
                        String[] split = null;
                        // Read one line
                        while((line = br.readLine()) != null){
                            split = line.split(",");
                        }
                        int x = 0;
                        for (String s : split) {
                            weight.set(x++, Float.parseFloat(s));
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

            }
        }
    }

    private boolean isFlat(jflow.layers.internal.Layer layer) {
        if (layer instanceof Dense) {
            return true;
        }
        return false;
    }

    /**
     * Print a model summary in the terminal.
     */
    public Sequential summary() {
        // Run an empty JMatrix through the model
        TrainableLayer first = (TrainableLayer)layers.getFirst();
        if (isFlat(first)) {
            JMatrix empty = new JMatrix(1, first.getInputShape()[0], 1, 1);
            forward(empty, false);
        } else {
            JMatrix empty = new JMatrix(1, first.getInputShape()[0], 
                first.getInputShape()[1], first.getInputShape()[2]);
            forward(empty, false);
        }

        // Sum trainable paramters
        int trainableParameters = 0;
        for (jflow.layers.internal.Layer l : layers) {
            trainableParameters += l.numTrainableParameters();
        }
        // Declare the size of each column
        int spacesType = 40;
        int spacesShape = 30;
        int spacesParam = 20;
        // Display the layer names and types
        String[][] layerTypes = new String[layers.size() + 1][2];
        layerTypes[0][0] = "Layer";
        layerTypes[0][1] = "type";
        
        for (int i = 0; i < layers.size(); i++) {
            String type = layers.get(i).getClass().getSimpleName();

            String name = layers.get(i).getName();
            layerTypes[i + 1][0] = name;
            layerTypes[i + 1][1] = type;
        }

        String[] shapes = new String[layers.size() + 1];
        shapes[0] = "Output Shape";
        for (int i = 0; i < layers.size(); i++) {
            int[] outputShape = layers.get(i).getOutputShape();
            String shapeAsString = "";
            for (int j = 1; j < outputShape.length; j++) {
                shapeAsString += outputShape[j];
                if (j != outputShape.length - 1) {
                    shapeAsString += ",";
                }
            }
            shapeAsString += ")";
            shapes[i + 1] = shapeAsString;
        }
        String[] params = new String[layers.size() + 1];
        params[0] = "Param #";
        for (int i = 0; i < layers.size(); i++) {
            params[i + 1] = String.valueOf(layers.get(i).numTrainableParameters());
        }

        String title = "\033[1;37m Model Summary";
        if (name != null) {
            title += " (" + name + ")";
        }
        title += "\033[0m\033[94m";
        System.out.println("\n" + title);
        System.out.print("╭");
        for (int i = 0; i < spacesType; i++) {
            System.out.print("─");
        }
        System.out.print("┬");
        for (int i = 0; i < spacesShape; i++) {
            System.out.print("─");
        }
        System.out.print("┬");
        for (int i = 0; i < spacesParam; i++) {
            System.out.print("─");
        }
        System.out.print("╮\033[0m\n");
        for (int line = 0; line < layerTypes.length; line++) {
            int numSpaces = spacesType - (layerTypes[line][0].length() + 
                layerTypes[line][1].length() + 4);

            System.out.print("\033[94m│ \033[0m");

            if (line == 0) { 
                System.out.print("\033[1;38;2;230;140;0m" + 
                    layerTypes[line][0] + "\033[0m \033[1;37m(" + layerTypes[line][1] + ")" + "\033[0m");
            } else {
                System.out.print("\033[38;2;255;165;0m" + 
                    layerTypes[line][0] + "\033[0m \033[37m(" + layerTypes[line][1] + ")" + "\033[0m");
            }
            
            for (int i = 0; i < numSpaces; i++) {
                System.out.print(" ");
            }

            
            System.out.print("\033[94m│ \033[0m");
            if (line == 0) {
                numSpaces = spacesShape - shapes[line].length() - 1;
                System.out.print("\033[1;37m");
                System.out.print(shapes[line]);
            } else {
                numSpaces = spacesShape - shapes[line].length() - 7;
                System.out.print("\033[38;2;0;153;153m(None\033[37m," + shapes[line] + "\033[0m");
            }
            
            for (int i = 0; i < numSpaces; i++) {
                System.out.print(" ");
            }

            numSpaces = spacesParam - params[line].length() - 1;
            System.out.print("\033[94m│ \033[0m");;
            for (int i = 0; i < numSpaces; i++) {
                System.out.print(" ");
            }
            if (line == 0) {
                System.out.print("\033[1;37m");
            } else {
                System.out.print("\033[37m");
            }
            System.out.print(params[line]);
            System.out.print("\033[94m│\033[0m\n");

            if (line == layerTypes.length - 1) {
                System.out.print("\033[94m╰");
            } else {
                System.out.print("\033[94m├");
            }
            
            for (int i = 0; i < spacesType; i++) {
                System.out.print("─");
            }
            if (line == layerTypes.length - 1) {
                System.out.print("┴");
            } else {
                System.out.print("┼");
            }
            for (int i = 0; i < spacesShape; i++) {
                System.out.print("─");
            }
            if (line == layerTypes.length - 1) {
                System.out.print("┴");
            } else {
                System.out.print("┼");
            }
            
            for (int i = 0; i < spacesParam; i++) {
                System.out.print("─");
            }
            if (line == layerTypes.length - 1) {
                System.out.print("╯");
            }
            else {
                System.out.print("┤");
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
        System.out.println("\033[1;38;2;230;140;0mTotal params: \033[1;37m" + parameters);
        System.out.println("\033[1;38;2;230;140;0mTrainable params: \033[1;37m" + parameters);
        System.out.println("\033[1;38;2;230;140;0mNon-trainable params: \033[1;37m" + 0 + "\033[0m");

        return this;
    }
}
