import java.io.IOException;
import java.util.function.Function;

import JFlow.Layers.*;
import JFlow.data.*;
import JFlow.utils.Metrics;

/*
 * In this demo, we train a convolutional neural network 
 * to distinguish trucks from automobiles, using the Cifar 10 dataset.
 */
public class CNNDemo {

    /*
     * Create a function that returns a double: 
     * learning rate from an integer: epoch.
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

    public static void main(String[] args) throws IOException {
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
        transform.normalizeTanH(); // Normalize to [-1, 1]

        // Apply the transform to images in the dataloader
        loader.applyTransform(transform);

        // Shuffle and split with a set seed
        loader.setSeed(42);
        loader.trainTestSplit(0.99);

        // Use augmentations for greater variation
        Transform augmentations = new Transform();
        augmentations.randomFlip();
        augmentations.randomBrightness();

        // Only applies augmentations to train images
        loader.applyAugmentations(augmentations);

        // Set batch size
        loader.batch(64);


        int numClasses = 2; // Binary classification
        int colorChannels = 3; // RGB images
        int imageSize = 32;

        // Initialize the model
        Sequential model = new Sequential();

        model.setInputShape(colorChannels, imageSize, imageSize); // Set input shape for convolutional layers

        // Block 1
        model.add(Layers.Conv2D(32, 3, "same_padding")); // (numFilters, filterSize, padding type)
        model.add(Layers.BatchNorm()); // Stabilize activations. Highly recommended
        model.add(Layers.LeakyReLU(0.01)); // Small leak to prevent dead neurons

        model.add(Layers.Conv2D(32, 3, "same_padding")); 
        model.add(Layers.BatchNorm());
        model.add(Layers.LeakyReLU(0.01)); 

        model.add(Layers.MaxPool2D(2, 2)); // (poolSize, stride)

        // Block 2
        model.add(Layers.Conv2D(64, 3, "same_padding"));
        model.add(Layers.BatchNorm());
        model.add(Layers.LeakyReLU(0.01));

        model.add(Layers.Conv2D(64, 3, "same_padding"));
        model.add(Layers.BatchNorm());
        model.add(Layers.LeakyReLU(0.01));
        
        model.add(Layers.MaxPool2D(2, 2));

        // Block 3
        model.add(Layers.Conv2D(128, 3, "same_padding"));
        model.add(Layers.BatchNorm());
        model.add(Layers.LeakyReLU(0.01));

        model.add(Layers.Conv2D(128, 3, "same_padding"));
        model.add(Layers.BatchNorm());
        model.add(Layers.LeakyReLU(0.01));
        
        model.add(Layers.MaxPool2D(2, 2));

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
            model.saveWeights("Cifar10 CNN Cars vs Trucks 2"); // Save weights to .txt files
        }
    }
}