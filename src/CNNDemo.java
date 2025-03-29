import java.io.IOException;
import JFlow.Layers.Layers;
import JFlow.Layers.Sequential;
import JFlow.data.Dataloader;
import JFlow.data.Transform;
import JFlow.utils.Metrics;

/*
 * In this demo, we train a convolutional neural network 
 * to distinguish cars from trucks, using the Cifar 10 dataset.
 */
public class CNNDemo {
    public static void main(String[] args) throws IOException {
        // Initialize the dataloader
        Dataloader loader = new Dataloader();

        // Declare labels to use a train labels reference csv, we only want cars and trucks
        String[] labelsToKeep = {"automobile","truck"};

        // Load a certain percentage of images from a directory with labels from a train labels reference csv, grayscale = false
        loader.loadFromDirectory("datasets/cifar10", labelsToKeep, "datasets/CifarTrainLabels.csv", 1.0, false);
        // loader.loadFromCSV("datasets/mouse_drawn_digits_2.txt", true, 1.0);


        // Initialize the transform
        Transform transform = new Transform();
        transform.normalizeTanH(); // Normalize to [-1, 1]

        // Apply the transform to images in the dataloader
        loader.applyTransform(transform);

        // Shuffle and split with a set seed
        loader.setSeed(42);
        loader.trainTestSplit(0.95);

        // Declare batch size
        loader.batch(32);


        int numClasses = 2; // Binary classification
        int colorChannels = 3; // RGB images
        int imageSize = 32;

        // Initialize the model
        Sequential model = new Sequential();

        model.add(Layers.Conv2D(64, colorChannels, 3, "same_padding")); // (numFilters, inputChannels, filterSize, padding)
        model.add(Layers.LeakyReLU(0.1));
        model.add(Layers.MaxPool2D(2, 2)); // (poolSize, stride)

        model.add(Layers.Conv2D(128, 64, 3, "same_padding"));
        model.add(Layers.LeakyReLU(0.1));
        model.add(Layers.MaxPool2D(2, 2)); 

        model.add(Layers.Dense((imageSize / 4) * (imageSize / 4) * 128, 256)); // (inputSize, outputSize)
        model.add(Layers.Dropout(0.3));
        model.add(Layers.LeakyReLU(0.1));

        model.add(Layers.Dense(256, numClasses));
        model.add(Layers.Softmax());


    // load trained weights
        // model.loadWeights("Cifar CNN Trucks vs Automobiles 64.128");


        // Use the Metrics class for model metrics
        double oldAccuracy = Metrics.getAccuracy(model.predict(loader.getTestImages()), loader.getTestLabels());

        /*
         * Automatically train the model on batches in the dataloader.
         * Specify number of epochs and learning rate.
         */
        model.train(loader, 10, 0.01);

        int[] predictions = model.predict(loader.getTestImages());

        // Display a confusion matrix in a new JFrame
        Metrics.displayConfusionMatrix(predictions, loader.getTestLabels());

        double newAccuracy = Metrics.getAccuracy(predictions, loader.getTestLabels());
        System.out.println("Test accuracy:" + newAccuracy);

        if (newAccuracy > oldAccuracy) {
            model.saveWeights("Cifar CNN Trucks vs Automobiles 64.128"); // Save weights to .txt files
        }
    }
}