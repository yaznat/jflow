package demos;
import jflow.data.*;
import jflow.model.*;
import jflow.utils.JPlot;
import jflow.utils.Metrics;

// Static import for cleaner UI
import static jflow.model.Builder.*;

/**
 * Demo to train a neural network on the MNIST dataset.
 * Reaches ~97% test accuracy after 10 epochs.
 */
public class NN {
    public static void main(String[] args) {
        // training constants
        final int BATCH_SIZE = 64;
        final double VAL_PERCENT = 0.05;
        final double TEST_PERCENT = 0.05;
        
        // MNIST constants
        final int NUM_CLASSES = 10;
        final int FLAT_IMAGE_SIZE = 784;

        // Initialize the dataloader
        Dataloader loader = new Dataloader();

        // Load data
        loader.loadFromCSV("datasets/MNIST.csv", true, 1.0);

        // Prepare the transform and apply normalization
        Transform transform = new Transform()
            .normalizeSigmoid();

        loader.applyTransform(transform);

        // Prepare data for training
        loader.setSeed(42);
        loader.valTestSplit(VAL_PERCENT, TEST_PERCENT);
        loader.batch(BATCH_SIZE); 

        // Visualize a random training image
        JPlot.displayImage(loader.getBatches()
            .get(0).get((int)(Math.random() * BATCH_SIZE)), 20);

        
        // Build the model
        Sequential model = new Sequential("MNIST_neural_network")
            .add(Dense(128, InputShape(FLAT_IMAGE_SIZE)))
            .add(Mish())

            .add(Dense(64))
            .add(Mish())
            .add(Dropout(0.3))

            .add(Dense(NUM_CLASSES))
            .add(Softmax())

            .summary();
        
    // load trained weights
        // model.loadWeights("saved_weights/MNIST NN"); 

    // Try out different optimizers
        // model.compile(SGD(0.1, 0.9, true));
        // model.compile(AdaGrad(0.01));
        // model.compile(RMSprop(0.001, 0.9, 1e-8, 0.9));
        model.compile(Adam(0.01));

        // Train the model
        model.train(loader, 10, ModelCheckpoint("val_loss", "saved_weights/MNIST NN"));

        // Evaluate the model
        int[] predictions = model.predict(loader.getTestImages());

        Metrics.displayConfusionMatrix(predictions, loader.getTestLabels());

        double newAccuracy = Metrics.getAccuracy(predictions, loader.getTestLabels());
        System.out.println("Test accuracy:" + newAccuracy);

    }
}