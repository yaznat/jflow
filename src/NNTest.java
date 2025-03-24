import java.io.IOException;
import JFlow.Layers.Layers;
import JFlow.Layers.Sequential;
import JFlow.data.Dataloader;
import JFlow.data.Transform;
import JFlow.utils.JPlot;

public class NNTest {
    public static void main(String[] args) throws IOException {
        Transform transform = new Transform();
        transform.normalizeSigmoid();

        Dataloader loader = new Dataloader();

        // loader.loadFromCSV("datasets/mouse_drawn_digits_2.txt", true, 1.0);
        loader.loadFromCSV("datasets/MNIST.csv", true, 1.0);
        // loader.addImagesWithLabel("datasets/cifar10-64/class0", 0, 0.3);
        // loader.addImagesWithLabel("datasets/cifar10-64/class1", 1, 0.3);
        // loader.addImagesWithLabel("datasets/cifar10-64/class2", 2, 0.3);
        // loader.addImagesWithLabel("datasets/cifar10-64/class3", 3, 0.3);
        // loader.addImagesWithLabel("datasets/cifar10-64/class4", 4, 0.3);
        // loader.addImagesWithLabel("datasets/cifar10-64/class5", 5, 0.3);
        // loader.addImagesWithLabel("datasets/cifar10-64/class6", 6, 0.3);
        // loader.addImagesWithLabel("datasets/cifar10-64/class7", 7, 0.3);
        // loader.addImagesWithLabel("datasets/cifar10-64/class8", 8, 0.3);
        // loader.addImagesWithLabel("datasets/cifar10-64/class9", 9, 0.3);

        loader.applyTransform(transform);

        loader.setSeed(42);
        loader.trainTestSplit(0.95);



        loader.batch(64);

        int numClasses = 10;

        int colorChannels = 1;

        Sequential model = new Sequential();

        model.add(Layers.Dense(784, 32));
        model.add(Layers.LeakyReLU(0.1));

        // model.add(Layers.Dense(28, 14));
        // model.add(Layers.LeakyReLU(0.1));

        // model.add(Layers.Dense(32, 16));
        // model.add(Layers.LeakyReLU(0.1));

        model.add(Layers.Dense(32, numClasses));
        model.add(Layers.Softmax());

        // model.loadWeights("Calculator NN 32.16");


        // model.setDebug(true);

        double oldAccuracy = model.getAccuracy(model.predict(loader.getTestImagesFlat()), loader.getTestLabels());


        model.train(loader, 10, 0.25);

        int[] predictions = model.predict(loader.getTestImagesFlat());

        JPlot.displayConfusionMatrix(predictions, loader.getTestLabels());

        double newAccuracy = model.getAccuracy(predictions, loader.getTestLabels());
        System.out.println("Test accuracy:" + newAccuracy);

        // if (newAccuracy > oldAccuracy) {
        //     model.saveWeights("Calculator NN 32.16");
        // }
    }
}