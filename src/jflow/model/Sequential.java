package jflow.model;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.stream.IntStream;

import jflow.data.*;
import jflow.layers.Dense;
import jflow.layers.Embedding;
import jflow.layers.Sigmoid;
import jflow.layers.templates.TrainableLayer;
import jflow.utils.Callbacks;
import jflow.utils.Metrics;



// The sequential object represents a model
public class Sequential{
    private ArrayList<jflow.model.Layer> layers = new ArrayList<>();
    private int numClasses = -1;
    private static int sequentialCount;
    private int modelNum;
    private String name = null;
    private boolean debugMode;
    private Optimizer optimizer;
    private int[] inputShape;
    private HashMap<String, JMatrix[]> layerGradients = new HashMap<>();
    private HashMap<String, Integer> layerCounts = new HashMap<>();

    private final String TEAL = "\033[38;2;0;153;153;1m";
    private final String YELLOW = "\033[38;2;222;197;15m";
    private final String ORANGE = "\033[38;2;255;165;1m";
    private final String BLUE = "\033[94m";
    private final String WHITE = "\033[37m";
    private final String GREEN = "\033[38;2;0;204;0m";
    private final String RED = "\033[38;2;255;0;0m";
    private final String BOLD = "\033[1m";
    private final String RESET = "\033[0m";

    /**
     * Initializes an empty Sequential model.
     */
    public Sequential(){
        modelNum = sequentialCount++;
    }

    /**
     * Initializes an empty Sequential model.
     */
    public Sequential(String name){
        this.name = name;
        modelNum = sequentialCount++;
    }
    /**
     * Add a layer to the model.
     * @param layer A JFlow Layer.
     */
    public Sequential add(Layer layer) {
        String name = layer.getName();
        // Update layer count in the hashmap
        layerCounts.put(name, layerCounts.getOrDefault(name, 0) + 1);

        if (!layer.isInternal()) {
            // Link non-internal layers
            if (layers.isEmpty() || onlyFunctionalLayers()) {
                // First layer in the model
                if (inputShape != null) {
                    layer.setInputShape(inputShape);
                }
            } else {
                // Find appropriate previous layer for connection
                // If the last layer was a functional layer, connect to it directly
                if (layers.getLast() instanceof FunctionalLayer) {
                    layer.setPreviousLayer(layers.getLast());
                    layers.getLast().setNextLayer(layer);
                } else {
                    // Otherwise find the last non-internal layer
                    Layer previousNonInternalLayer = null;
                    for (int i = layers.size() - 1; i >= 0; i--) {
                        if (!layers.get(i).isInternal()) {
                            previousNonInternalLayer = layers.get(i);
                            break;
                        }
                    }
                    if (previousNonInternalLayer != null) {
                        layer.setPreviousLayer(previousNonInternalLayer);
                        previousNonInternalLayer.setNextLayer(layer);
                    }
                }
            }
        }
        
        // Add the layer to layers
        layers.add(layer);

        // Build the layer after setting connections
        layer.build(layerCounts.get(name));
        
        // Special handling for FunctionalLayer
        if (layer instanceof FunctionalLayer) {
            processFunctionalLayer((FunctionalLayer) layer);
        }

      
        
        return this;
    }

    // Checks if layers contains only FunctionalLayers 
    private boolean onlyFunctionalLayers() {
        if (layers.size() == 0) {
            return false;
        }
        for (Layer l : layers) {
            if (!(l instanceof FunctionalLayer)) {
                return false;
            }
        }
        return true;
    }

    private void processFunctionalLayer(FunctionalLayer functionalLayer) {
        Layer[] internalLayers = functionalLayer.getLayers();
        
        if (internalLayers != null && internalLayers.length > 0) {
            // Find the appropriate previous layer to connect the first internal layer
            Layer previousLayerForConnection = findPreviousLayerForInternalLayer(functionalLayer);
            
            if (previousLayerForConnection != null) {
                internalLayers[0].setPreviousLayer(previousLayerForConnection);
                previousLayerForConnection.setNextLayer(internalLayers[0]);
            }
            
            // Connect internal layers to each other
            for (int i = 1; i < internalLayers.length; i++) {
                internalLayers[i].setPreviousLayer(internalLayers[i - 1]);
                internalLayers[i - 1].setNextLayer(internalLayers[i]);
            }
            
            // Process each internal layer
            for (Layer internalLayer : internalLayers) {
                String internalName = internalLayer.getName();
                layerCounts.put(internalName, layerCounts.getOrDefault(internalName, 0) + 1);
                internalLayer.build(layerCounts.get(internalName));
                layers.add(internalLayer);
                
                // Recursively process nested functional layers
                if (internalLayer instanceof FunctionalLayer) {
                    processFunctionalLayer((FunctionalLayer) internalLayer);
                }
            }
        }
    }
    private Layer findPreviousLayerForInternalLayer(FunctionalLayer functionalLayer) {
        // Find functional layer's index
        int functionalLayerIndex = -1;
        for (int i = 0; i < layers.size(); i++) {
            if (layers.get(i) == functionalLayer) {
                functionalLayerIndex = i;
                break;
            }
        }
        
        if (functionalLayerIndex < 0) {
            return null; // Functional layer not found in layers list
        }
        
        // Find the previous layer that the first internal layer should connect to
        // This should be the last non-internal layer before this functional layer
        // or a layer with a different enclosing functional layer
        
        for (int i = functionalLayerIndex - 1; i >= 0; i--) {
            Layer candidate = layers.get(i);
            
            // If the candidate is not internal, it's a valid connection
            if (!candidate.isInternal()) {
                return candidate;
            }
            
            // If the candidate has a different enclosing layer than our functional layer,
            // it could be a valid connection
            Layer candidateEnclosing = candidate.getEnclosingLayer();
            if (candidateEnclosing != null && !candidateEnclosing.equals(functionalLayer)) {
                // We need to check if the candidate's enclosing layer is not contained within our functional layer
                // This is to prevent connecting to layers in deeper nested functional layers
                boolean isValid = true;
                Layer checkLayer = candidateEnclosing;
                
                // Traverse up the enclosing chain to ensure we're not connecting to a deeper nested layer
                while (checkLayer != null) {
                    if (checkLayer.equals(functionalLayer)) {
                        isValid = false;
                        break;
                    }
                    checkLayer = checkLayer.getEnclosingLayer();
                }
                
                if (isValid) {
                    return candidate;
                }
            }
        }
        
        return null;
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
     * @param optimizer The desired optimizer.
     */
    public Optimizer compile(Optimizer optimizer) {
        setOptimizer(optimizer);
        return optimizer;
    }

    // Initialize each trainable layer in the optimizer
    private void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
        for (jflow.model.Layer l : layers) {
            if (l instanceof TrainableLayer) {
                TrainableLayer trainable = (TrainableLayer)l;
                optimizer.initializeLayer(trainable);
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
     * Disable gradient storage for very large models. <p>
     * 
     * Gradient storage is enabled by default, keeping a record of the most recent forward output
     * and backward gradient for each layer. This behavior is crucial for gradient debugging, 
     * as well as allowing the user to manually access these values from a layer. In the case of 
     * very large models, it may be necessary to disable gradient storage for the sake of memory.
     */
    public Sequential disableGradientStorage() {
        for (Layer l : layers) {
            l.disableGradientStorage();
        }
        return this;
    }
    /**
     * Train the model.
     * @param loader                A Dataloader containing train images.
     * @param epochs                The number of epochs to train.
     */
    public void train(Dataloader loader, int epochs) {
        runTraining(loader, epochs, null);
    }

    /**
     * Train the model.
     * @param loader                A Dataloader containing train images.
     * @param epochs                The number of epochs to train.
     * @param savePath              The path for saving checkpoints. Either:
     *                              <ul> <li> saves every epoch that validation   
     */
    public void train(Dataloader loader, int epochs, ModelCheckpoint checkpoint) {
        runTraining(loader, epochs, checkpoint);
    }

    private void runTraining(Dataloader loader, int epochs,
        ModelCheckpoint checkpoint) {
        // Ensure there is an optimizer
        if (optimizer == null) {
            setOptimizer(new SGD(0.01)); // Simplest possible optimizer
        }
        // Store values for metric tracking
        double prevTrainAccuracy = 0;
        double bestTrainAccuracy = 0;

        double bestTrainLoss = Double.POSITIVE_INFINITY;

        double prevValAccuracy = 0;
        double bestValAccuracy = 0;

        double prevValLoss = Double.POSITIVE_INFINITY;
        double bestValLoss = Double.POSITIVE_INFINITY;

        // Print training header
        String name = (this.name == null) ? "sequential_" + modelNum : this.name;
        Callbacks.printTrainingHeader(name);
        // Prepare validation data
        JMatrix valData = null;
        int[] valLabels = null;
        boolean useValSet = false;

        if (loader.imageBreakdown().get("val") > 0) {
            valData = loader.getValImages();
            valLabels = loader.getValLabels();
            useValSet = true;
        }
        
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

                int[] predictions = getPredictions(output);

                accuracy += Metrics.getAccuracy(predictions, yBatch);

                totalLoss += crossEntropyLoss(output, yBatch);

                long batchTime = System.nanoTime();
                long timeSinceStart = batchTime - startTime;

                LinkedHashMap<String, Double> lossReport = new LinkedHashMap<>();
                lossReport.put("Loss", totalLoss / (batch + 1));

                if (!debugMode) {
                    Callbacks.printProgressCallback("Epoch", epoch, epochs, "Batch", batch + 1, numBatches,
                        timeSinceStart, lossReport);
                }
            }
            Double trainLoss = totalLoss / numBatches;

            // Report train accuracy
            String report = BLUE + "    Training Accuracy: " + RESET;
            // Convert to percentage
            double trainAccuracy = accuracy / loader.numBatches();
            String trainPercentage = accuracyToPercentage(trainAccuracy);

            // Warn if performance declines with RED
            if (trainAccuracy > prevTrainAccuracy) {
                report += GREEN;
            } else {
                report += RED;
            }
            prevTrainAccuracy = trainAccuracy;

            report += trainPercentage + RESET;

            double valLoss = Double.POSITIVE_INFINITY;
            double valAccuracy = 0;
            if (useValSet) {
                // Report validation accuracy
                report += BLUE + "\n    Validation Accuracy: " + RESET;
                // Test on the val set
                int[] valPredictions = predict(valData);
                valAccuracy = Metrics.getAccuracy(valPredictions, valLabels);
                String valPercentage = accuracyToPercentage(valAccuracy);

                // Warn if performance declines with RED
                if (valAccuracy > prevValAccuracy) {
                    report += GREEN;
                } else {
                    report += RED;
                }
                prevValAccuracy = valAccuracy;

                report += valPercentage + RESET;

                // Report validation loss
                report += BLUE + "\n    Validation Loss: ";
                valLoss = crossEntropyLoss(layers.getLast().getOutput(), valLabels);

                // Warn if performance declines with RED
                if (valLoss < prevValLoss) {
                    report += GREEN;
                } else {
                    report += RED;
                }
                prevValLoss = valLoss;

                report += capDouble(valLoss, 8) + RESET;
            }
            System.out.println("\n" + report);
            if (checkpoint == null) {
                System.out.println("");
            } else {
                String reportName = null;
                String val1 = null;
                String val2 = null;
                boolean improved = false;
                // check the given metric
                switch (checkpoint.getMetric()) {
                    case "val_loss":
                        reportName = "Validation loss";
                        val1 = capDouble(bestValLoss, 8);
                        val2 = capDouble(valLoss, 8);
                        if (valLoss < bestValLoss) {
                            improved = true;
                            bestValLoss = valLoss;
                        } else {
                            improved = false;
                        }
                        break;
                    case "val_accuracy":
                        reportName = "Validation accuracy";
                        val1 = capDouble(bestValAccuracy, 8);
                        val2 = capDouble(valAccuracy, 8);
                        if (valAccuracy > bestValAccuracy) {
                            improved = true;
                            bestValAccuracy = valAccuracy;
                        } else {
                            improved = false;
                        }
                        break;
                    case "train_loss":
                        reportName = "Train loss";
                        val1 = capDouble(bestTrainLoss, 8);
                        val2 = capDouble(trainLoss, 8);
                        if (trainLoss < bestTrainLoss) {
                            improved = true;
                            bestTrainLoss = trainLoss;
                        } else {
                            improved = false;
                        }
                        break;
                    case "train_accuracy":
                        reportName = "Train accuracy";
                        val1 = capDouble(bestTrainAccuracy, 8);
                        val2 = capDouble(trainAccuracy, 8);
                        if (trainAccuracy > bestTrainAccuracy) {
                            improved = true;
                            bestTrainAccuracy = trainAccuracy;
                        } else {
                            improved = false;
                        }
                        break;
                }
                if (improved) {
                    System.out.println(
                        WHITE + reportName + " improved from "
                        + BLUE + val1 + 
                        WHITE + " to " + BLUE + val2 +
                        WHITE + ". Saving model to " + BLUE + 
                        checkpoint.getSavePath() + RESET + "\n"
                    );
                    internalSaveWeights(checkpoint.getSavePath(), false);
                } else {
                    System.out.println(
                        WHITE + reportName + " did not improve from "
                        + BLUE + val1 + RESET + "\n"
                    );
                }
            }
        }
    }

    // Converts accuracy to percentage
    private String accuracyToPercentage(double accuracy) {
        // Cap to 4 significant figures
        return capDouble(accuracy * 100, 5) + "%";
    }
    // Convert a double to a String with given length
    private String capDouble(double number, int length) {
        String numAsString = String.valueOf(number);
        // Avoid StingIndexOutOfBoundsException
        while (numAsString.length() < length) {
            numAsString += "0";
        }
        return numAsString.substring(0, length);
    }

    /**
     * Predict class labels on batched image data in a JMatrix.
     * @param images                    a JMatrix of images in the shape (N, channels, height, width).
     * @return                      predicted class labels in the range [0, numClasses].
     */
    public int[] predict(JMatrix images) {
        // Forward pass
        JMatrix output = forward(images, false);

        // Get predictions
        return getPredictions(output);
        
    }

    // Internal helper to convert output to predictions
    private int[] getPredictions(JMatrix output) {
        int batchSize = output.channels();
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
        JMatrix output = images;
        for (int i = 0; i < layers.size(); i++) {
            if (!layers.get(i).isInternal()) {
                output = layers.get(i).forward(output, training);
            }
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
        JMatrix gradient = yTrue;
        for (int i = layers.size() - 1; i >= 0; i--) {
            if (!layers.get(i).isInternal()) {
                gradient = layers.get(i).backward(gradient);
            }
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
     * Save weights to binary files in a directory.
     * @param path               The name of the directory to store files in.
     */
    public void saveWeights(String path) {
        internalSaveWeights(path, true);
    }
    public void internalSaveWeights(String path, boolean printReport) {
        // Save trainable layer weights
        IntStream.range(0, layers.size())
            .parallel()
            .forEach(i -> {
                jflow.model.Layer l = layers.get(i);
                if (l instanceof TrainableLayer trainable) {
                    for (JMatrix weight : trainable.getWeights()) {
                        String filePath = path + "/" + trainable.getName() + "_" + weight.getName() + ".bin";
                        saveWeightToBinary(filePath, weight);
                    }
                }
            });
    
        // Save optimizer time steps
        if (optimizer != null) {
            if (optimizer instanceof Adam adam) {
                String timestepPath = path + "/" + optimizer.getName() + "/timesteps.bin";
                try {
                    Path dir = Paths.get(path + "/" + optimizer.getName());
                    Files.createDirectories(dir);
            
                    try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(timestepPath))) {
                        dos.writeLong(adam.getTimeSteps()); // Write 8-byte long
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
    
            // Save optimizer weights
            JMatrix[] optWeights = optimizer.getWeights();
            IntStream.range(0, optWeights.length)
                .parallel()
                .forEach(i -> {
                    JMatrix weight = optWeights[i];
                    String filePath = path + "/" + optimizer.getName() + "/" + weight.getName() + ".bin";
                    saveWeightToBinary(filePath, weight);
                });
        }
    
        if (printReport) {
            System.out.println("Weights saved to " + path);
        }
    }

    // Helper method to write weight values to binary
    private void saveWeightToBinary(String filePath, JMatrix weight) {
        try {
            Path dir = Paths.get(filePath).getParent();
            if (dir != null) Files.createDirectories(dir);

            try (DataOutputStream dos = new DataOutputStream(
                    new BufferedOutputStream(
                        new FileOutputStream(filePath)))) {
                for (int i = 0; i < weight.size(); i++) {
                    dos.writeFloat(weight.get(i)); // Write 4 bytes per float
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    /**
     * Load weights from binary files in a directory.
     * @param path               The location of the directory to load files from.
     */
    public void loadWeights(String path) {
        // Load all trainable layer weights
        IntStream.range(0, layers.size())
            .parallel()
            .forEach(i -> {
                jflow.model.Layer l = layers.get(i);
                if (l instanceof TrainableLayer trainable) {
                    JMatrix[] weights = trainable.getWeights();
                    for (JMatrix weight : weights) {
                        String filePath = path + "/" + trainable.getName() + "_" + weight.getName() + ".bin";
                        loadWeightFromBinary(filePath, weight);
                    }
                }
            });

        if (optimizer != null) {
            // Load timestep
            if (optimizer instanceof Adam adam) {
                String timestepPath = path + "/" + optimizer.getName() + "/timesteps.bin";
                try (DataInputStream dis = new DataInputStream(new FileInputStream(timestepPath))) {
                    adam.setTimeSteps(dis.readLong());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            // Load optimizer weights
            JMatrix[] weights = optimizer.getWeights();
            IntStream.range(0, weights.length)
                .parallel()
                .forEach(i -> {
                    JMatrix weight = weights[i];
                    String filePath = path + "/" + optimizer.getName() + "/" + weight.getName() + ".bin";
                    loadWeightFromBinary(filePath, weight);
                });
        }
    }

    // Helper method to read a binary file into a JMatrix
    private void loadWeightFromBinary(String filePath, JMatrix weight) {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            for (int i = 0; i < weight.size(); i++) {
                weight.set(i, dis.readFloat());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    


    private boolean isFlat(Layer layer) {
        return layer instanceof Dense || layer instanceof Embedding;
    }

    /**
     * Print a model summary in the terminal.
     */
    public Sequential summary() {
        // find the first Trainable Layer
        int layerIndex = 0;
        Layer finder = layers.get(layerIndex++);
        while (! (finder instanceof TrainableLayer)) {
            finder = layers.get(layerIndex++);
        }
        TrainableLayer first = (TrainableLayer)finder;
        // Run an empty JMatrix through the model
        if (isFlat(first)) {
            // Run a flat batch of 1 through the model
            if (first instanceof Embedding) {
                JMatrix empty = new JMatrix(1, 1, 1, 1);
                forward(empty, false);
            } else {
                // Dense input shape is fixed
                JMatrix empty = new JMatrix(1, first.getInputShape()[0], 1, 1);
                forward(empty, false);
            }
        } else {
            // Run a 4D batch of 1 through the model
            JMatrix empty = new JMatrix(1, first.getInputShape()[0], 
                first.getInputShape()[1], first.getInputShape()[2]);
            forward(empty, false);
        }
        // Count the total number of layers that aren't FunctionalLayers
        int numLayers = layers.size();
        for (Layer l : layers) {
            if (l instanceof FunctionalLayer) {
                numLayers--;
            }
        }

        // Sum trainable paramters
        int trainableParameters = 0;
        for (Layer l : layers) {
            trainableParameters += l.numTrainableParameters();
        }

        // Declare the size of each column
        int spacesType = 40;
        int spacesShape = 30;
        int spacesParam = 20;

        // Display the layer names and types
        String[][] layerTypes = new String[numLayers + 1][2];
        layerTypes[0][0] = "Layer";
        layerTypes[0][1] = "type";

        int targetIndex = 1; // Start after header
        for (Layer layer : layers) {
            if (layer instanceof FunctionalLayer) {
                continue; // Skip this layer
            }

            String type = layer.getClass().getSimpleName();
            String name = layer.getName();
            layerTypes[targetIndex][0] = name;
            layerTypes[targetIndex][1] = type;
            targetIndex++;
        }

        String[] shapes = new String[numLayers + 1];
        shapes[0] = "Output Shape";

        targetIndex = 1; // Start after header
        for (Layer layer : layers) {
            if (layer instanceof FunctionalLayer) {
                continue; // Skip this layer
            }

            int[] outputShape = layer.outputShape();
            String shapeAsString = "";

            for (int j = 1; j < outputShape.length; j++) {
                shapeAsString += outputShape[j];
                if (j != outputShape.length - 1) {
                    shapeAsString += ",";
                }
            }

            shapeAsString += ")";
            shapes[targetIndex++] = shapeAsString;
        }
        String[] params = new String[numLayers + 1];
        params[0] = "Param #";
        
        targetIndex = 1; // Start after header
        for (Layer layer : layers) {
            if (layer instanceof FunctionalLayer) {
                continue; // Skip this layer
            }
            params[targetIndex++] = String.valueOf(layer.numTrainableParameters());
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
