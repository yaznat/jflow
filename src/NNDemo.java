import java.io.IOException;
import JFlow.Layers.Layers;
import JFlow.Layers.Sequential;
import JFlow.data.Dataloader;
import JFlow.data.Transform;
import JFlow.utils.JPlot;

public class NNDemo {
    public static void main(String[] args) throws IOException {
        Dataloader loader = new Dataloader();

        loader.loadFromCSV("datasets/MNIST.csv", true, 1.0);

        Transform transform = new Transform();
        transform.normalizeSigmoid();

        loader.applyTransform(transform);

        loader.setSeed(42);
        loader.trainTestSplit(0.95);

        loader.batch(64);

        int numClasses = 10;

        Sequential model = new Sequential();

        model.add(Layers.Dense(784, 128));
        model.add(Layers.Dropout(0.3));
        model.add(Layers.ReLU());
        
        model.add(Layers.Dense(128, numClasses));
        model.add(Layers.Softmax());

        // model.loadWeights("MNIST NN 128");


        double oldAccuracy = model.getAccuracy(model.predict(loader.getTestImagesFlat()), loader.getTestLabels());

        model.train(loader, 10, 0.25);

        int[] predictions = model.predict(loader.getTestImagesFlat());

        JPlot.displayConfusionMatrix(predictions, loader.getTestLabels());

        double newAccuracy = model.getAccuracy(predictions, loader.getTestLabels());
        System.out.println("Test accuracy:" + newAccuracy);

        if (newAccuracy > oldAccuracy) {
            model.saveWeights("MNIST NN 128");
        }
    }
}