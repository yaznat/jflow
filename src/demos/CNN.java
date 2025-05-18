package demos;
import jflow.data.*;
import jflow.model.Sequential;
import jflow.utils.JPlot;
import jflow.utils.Metrics;

// Static import for cleaner UI
import static jflow.model.Builder.*;
/**
 * Demo to train a convolutional neural network (CNN) 
 * to classify trucks vs. automobiles using the CIFAR-10 dataset.
 */
public class CNN {
    // Helper method to add a block to the model
    private static void addConvBlock(Sequential model, int filters) {
        model
            .add(Conv2D(filters, 3, 1, "same_padding"))
            .add(Swish())
            .add(BatchNorm())
            .add(Dropout(0.25))

            .add(MaxPool2D(2, 2));

    }

    public static void main(String[] args) {
        // training constants
        final int BATCH_SIZE = 64;
        final double VAL_PERCENT = 0.02;
        final double TEST_PERCENT = 0.02;
        
        // Cifar10 constants
        final int COLOR_CHANNELS = 3; // RGB images
        final int IMAGE_SIZE = 32;
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
        loader.valTestSplit(VAL_PERCENT, TEST_PERCENT);
        loader.batch(BATCH_SIZE);

        // Visualize a random training image
        JPlot.displayImage(loader.getBatches()
            .get(0).get((int)(Math.random() * BATCH_SIZE)), 20);

        // Add simple augmentations to train images
        Transform augmentations = new Transform()
            .randomFlip();
            // .randomBrightness();

        loader.applyAugmentations(augmentations);


        // Build the model
        Sequential model = new Sequential("Cifar10_CNN");

        model.setInputShape(InputShape(COLOR_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)); 

        // Block 1
        addConvBlock(model, 32);

        // Block 2
        addConvBlock(model, 64);


        // Flatten and Dense layers
        model
            .add(Flatten())
            .add(Dense(64))
            .add(Swish())
            .add(Dropout(0.3))

            .add(Dense(1))
            .add(Sigmoid()) // Sigmoid for binary classification

            .summary();

    // Try out different optimizers
        // model.compile(SGD(0.01, 0.9, true));
        // model.compile(AdaGrad(0.01));
        // model.compile(RMSprop(0.001, 0.9, 1e-8, 0.9));
        model.compile(Adam(0.001));

    // Load trained weights
        // model.loadWeights("saved_weights/Cifar10 CNN Cars vs Trucks");

        // Train the model
        model.train(loader, 30, ModelCheckpoint(
            "val_loss", "saved_weights/Cifar10 CNN Cars vs Trucks"));

        // Evaluate the model
        int[] predictions = model.predict(loader.getTestImages());

        Metrics.displayConfusionMatrix(predictions, loader.getTestLabels());

        double newAccuracy = Metrics.getAccuracy(predictions, loader.getTestLabels());
        System.out.println("Test accuracy:" + newAccuracy);
    }
}