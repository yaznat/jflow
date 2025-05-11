import jflow.data.*;
import jflow.model.*;
import jflow.utils.Metrics;

/**
 * Demo to train a neural network on the MNIST dataset.
 * Reaches ~97% test accuracy after 10 epochs.
 */
public class NN extends Builder{
    public static void main(String[] args) {
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
        loader.trainTestSplit(0.95);
        loader.batch(64); 

        // MNIST constants
        final int numClasses = 10;
        final int flattenedImageSize = 784;

        // Build the model
        Sequential model = new Sequential()
            .add(Dense(128, InputShape(flattenedImageSize)))
            .add(Mish())

            .add(Dense(64))
            .add(Mish())
            .add(Dropout(0.3))


            .add(Dense(numClasses))
            .add(Softmax())

            .summary();
        
    // load trained weights
        // model.loadWeights("MNIST NN"); 


        model.compile(Adam(0.01));

        double oldAccuracy = Metrics.getAccuracy(model.predict(loader.getTestImagesFlat()), loader.getTestLabels());

        // Train the model
        model.train(loader, 10);

        // Evaluate the model
        int[] predictions = model.predict(loader.getTestImagesFlat());

        Metrics.displayConfusionMatrix(predictions, loader.getTestLabels());

        double newAccuracy = Metrics.getAccuracy(predictions, loader.getTestLabels());
        System.out.println("Test accuracy:" + newAccuracy);

        if (newAccuracy > oldAccuracy) {
            model.saveWeights("saved_weights/MNIST NN");
        }
    }
}