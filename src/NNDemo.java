import java.io.IOException;

import JFlow.Layers.*;
import JFlow.data.*;
import JFlow.utils.Metrics;

/*
 * In this demo, we train a simple neural network on the MNIST dataset.
 * Reaches ~97% test accuracy after 10 epochs.
 */
public class NNDemo {
    public static void main(String[] args) throws IOException {
        // Initialize the dataloader
        Dataloader loader = new Dataloader();

        /* 
         * Load flat images from a csv or txt file.
         * Indicate if labels are the first item in each row.
         * Specify the percent of all images that you want to load.
         */ 
        loader.loadFromCSV("datasets/MNIST.csv", true, 1.0);

        // Initialize the transform
        Transform transform = new Transform();
        transform.normalizeSigmoid(); // Normalize data to [0, 1]

        // Apply the transform to images in the dataloader
        loader.applyTransform(transform);

        // Shuffle and split with a set seed
        loader.setSeed(42);
        loader.trainTestSplit(0.95);

        // Declare the batch size
        loader.batch(64);

        int numClasses = 10;

        // Initialize the model
        Sequential model = new Sequential();

        model.add(Layers.Dense(784, 128)); // (Input size, output size)
        model.add(Layers.Dropout(0.3));
        model.add(Layers.ReLU());
        
        model.add(Layers.Dense(128, numClasses));
        model.add(Layers.Softmax());

    // load trained weights
        // model.loadWeights("MNIST NN 128"); 


        // Use the Metrics class for model metrics
        double oldAccuracy = Metrics.getAccuracy(model.predict(loader.getTestImagesFlat()), loader.getTestLabels());

        /*
         * Automatically train the model on batches in the dataloader.
         * Specify number of epochs and learning rate.
         */
        model.train(loader, 10, 0.25);

        int[] predictions = model.predict(loader.getTestImagesFlat());

        // Display a confusion matrix in a new JFrame
        Metrics.displayConfusionMatrix(predictions, loader.getTestLabels());

        double newAccuracy = Metrics.getAccuracy(predictions, loader.getTestLabels());
        System.out.println("Test accuracy:" + newAccuracy);

        if (newAccuracy > oldAccuracy) {
            model.saveWeights("MNIST NN 128"); // Save weights to .txt files
        }
    }
}