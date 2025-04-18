import java.util.function.Function;

import JFlow.Layers.*;
import JFlow.data.*;
import JFlow.utils.Metrics;

/**
 * Demo to train a convolutional neural network (CNN) 
 * to classify trucks vs. automobiles using the CIFAR-10 dataset.
 */
public class CNNDemo {

    /**
     * Learning rate scheduler
     * 
     * @return Returns a learning rate based on the current epoch:
     * <ul>
     *   <li><b>Epoch ≤ 10:</b> 0.01</li>
     *   <li><b>Epoch ≤ 20:</b> 0.003</li>
     *   <li><b>Epoch > 20:</b> 0.001</li>
     * </ul>
     */
    public static Function<Integer, Double> lrSchedule() {
        return epoch -> {
            if (epoch <= 10) {
                return 0.01;
            } else if (epoch <= 20) {
                return 0.003;
            } else {
                return 0.001;
            }
        };
    }

    // Helper method adds a block to the model
    private static void addConvBlock(Sequential model, int filters) {
        model.add(Layers.Conv2D(filters, 3, "same_padding")); // (filters, kernelSize, paddingType)
        model.add(Layers.BatchNorm()); // Stabilize activations. Highly recommended
        model.add(Layers.LeakyReLU(0.01)); // Small leak to prevent dead neurons
    
        model.add(Layers.Conv2D(filters, 3, "same_padding"));
        model.add(Layers.BatchNorm());
        model.add(Layers.LeakyReLU(0.01));
    
        model.add(Layers.MaxPool2D(2, 2)); // (poolSize, stride)
    }

    public static void main(String[] args) {
        // Initialize the dataloader
        Dataloader loader = new Dataloader();

        /*
         * When low memory mode is on, 
         * images are not kept loaded.
         * It takes slightly longer to
         * access them.
         */
        loader.setLowMemoryMode(true); // Conserve memory on a larger dataset

        // Declare labels to use a train labels reference csv, we only want cars and trucks
        String[] labelsToKeep = {"automobile","truck"};

        /*
         * Load a certain percent of images from a directory with 
         * labels from a train labels reference csv, grayscale = false
         */ 
        loader.loadFromDirectory("datasets/cifar10", labelsToKeep, 
            "datasets/CifarTrainLabels.csv", 1.0, false);


        // Initialize the transform
        Transform transform = new Transform();
        transform.normalizeTanh(); // Normalize to [-1, 1]

        // Apply the transform to images in the dataloader
        loader.applyTransform(transform);

        // Shuffle and split with a set seed
        loader.setSeed(42);
        loader.trainTestSplit(0.99);

        // Add basic augmentations to train images
        Transform augmentations = new Transform();
        augmentations.randomFlip();
        augmentations.randomBrightness();
        loader.applyAugmentations(augmentations);

        loader.batch(64); // Set batch size


        int numClasses = 2; // Binary classification
        int colorChannels = 3; // RGB images
        int imageSize = 32;

        // Initialize the model
        Sequential model = new Sequential();

        model.setInputShape(colorChannels, imageSize, imageSize); // (Channels, height, width)

        // Block 1
        addConvBlock(model, 32);

        // Block 2
        addConvBlock(model, 64);

        // Block 3
        addConvBlock(model, 128);

        // Flattened fully connected layers
        model.add(Layers.Dense((imageSize / 8) * (imageSize / 8) * 128, 128)); // (inputSize, outputSize)
        model.add(Layers.LeakyReLU(0.01));
        model.add(Layers.Dropout(0.5)); // Disable 50% of the neurons

        model.add(Layers.Dense(128, numClasses));
        model.add(Layers.Softmax());

        // Print a summary in the terminal
        model.summary();


    // load trained weights
        // model.loadWeights("Cifar10 CNN Cars vs Trucks");

        // Use the Metrics class for model metrics
        double oldAccuracy = Metrics.getAccuracy(model.predict(loader.getTestImages()), loader.getTestLabels());

        /*
         * Automatically train the model on batches in the dataloader,
         * passing in our learning rate function.
         */
        model.train(loader, 30, lrSchedule());

        int[] predictions = model.predict(loader.getTestImages());

        // Display a confusion matrix in a new JFrame
        Metrics.displayConfusionMatrix(predictions, loader.getTestLabels());

        double newAccuracy = Metrics.getAccuracy(predictions, loader.getTestLabels());
        System.out.println("Test accuracy:" + newAccuracy);

        if (newAccuracy > oldAccuracy) {
            model.saveWeights("Cifar10 CNN Cars vs Trucks"); // Save weights to .txt files
        }
    }
}