import jflow.data.*;
import jflow.model.Builder;
import jflow.model.Sequential;
import jflow.utils.Metrics;

/**
 * Demo to train a convolutional neural network (CNN) 
 * to classify trucks vs. automobiles using the CIFAR-10 dataset.
 */
public class CNN extends Builder{
    // Helper method to add a block to the model
    private static void addConvBlock(Sequential model, int filters) {
        model
            .add(Conv2D(filters, 3, 1, "same_padding"))
            .add(LeakyReLU(0.01))
            .add(BatchNorm())
            .add(Dropout(0.1))

            .add(MaxPool2D(2, 2));
    }

    public static void main(String[] args) {
        // Load data
        Dataloader loader = new Dataloader();
    // Use if necessary
        // loader.setLowMemoryMode(true); 

        /*
         * Declare labels to use a train labels reference csv.
         * We only want cars and trucks.
         */ 
        String[] labelsToKeep = {"automobile","truck"};

        loader.loadFromDirectory("datasets/cifar10", labelsToKeep, 
            "datasets/CifarTrainLabels.csv", 1.0, false);


        // Prepare the transform and apply normalization
        Transform transform = new Transform()
            .normalizeTanh(); 

        loader.applyTransform(transform);

        // Prepare data for training
        loader.setSeed(42);
        loader.trainTestSplit(0.99);
        loader.batch(64);

        // Add simple augmentations to train images
        Transform augmentations = new Transform()
            .randomFlip()
            .randomBrightness();

        loader.applyAugmentations(augmentations);

        // CIFAR-10 constants
        final int colorChannels = 3; // RGB images
        final int imageSize = 32;


        // Build the model
        Sequential model = new Sequential();

        model.setInputShape(InputShape(colorChannels, imageSize, imageSize)); 

        // Block 1
        addConvBlock(model, 32);

        // Block 2
        addConvBlock(model, 64);

        // Block 3
        addConvBlock(model, 128);


        // Flatten and Dense layers
        model
            .add(Flatten())
            .add(Dense(128))
            .add(LeakyReLU(0.01))
            .add(Dropout(0.3))

            .add(Dense(1))
            .add(Sigmoid()) // Sigmoid for binary classification

            .summary();


    // Load trained weights
        // model.loadWeights("saved_weights/Cifar10 CNN Cars vs Trucks");

        model.compile(Adam(0.001));


        double oldAccuracy = Metrics.getAccuracy(model.predict(loader.getTestImages()), loader.getTestLabels());

        System.out.println(oldAccuracy);
        
        // Train the model
        model.train(loader, 30);

        // Evaluate the model
        int[] predictions = model.predict(loader.getTestImages());

        Metrics.displayConfusionMatrix(predictions, loader.getTestLabels());

        double newAccuracy = Metrics.getAccuracy(predictions, loader.getTestLabels());
        System.out.println("Test accuracy:" + newAccuracy);

        if (newAccuracy > oldAccuracy) {
            model.saveWeights("saved_weights/Cifar10 CNN Cars vs Trucks");
        }
    }
}