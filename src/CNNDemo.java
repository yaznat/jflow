import java.io.IOException;
import JFlow.Layers.Layers;
import JFlow.Layers.Sequential;
import JFlow.data.Dataloader;
import JFlow.data.Transform;
import JFlow.utils.JPlot;

public class CNNDemo {
    public static void main(String[] args) throws IOException {
        Dataloader loader = new Dataloader();

        loader.addImagesWithLabel("datasets/cifar10-64/class0", 0, 0.3);
        loader.addImagesWithLabel("datasets/cifar10-64/class1", 1, 0.3);
        loader.addImagesWithLabel("datasets/cifar10-64/class2", 2, 0.3);
        loader.addImagesWithLabel("datasets/cifar10-64/class3", 3, 0.3);
        loader.addImagesWithLabel("datasets/cifar10-64/class4", 4, 0.3);
        loader.addImagesWithLabel("datasets/cifar10-64/class5", 5, 0.3);
        loader.addImagesWithLabel("datasets/cifar10-64/class6", 6, 0.3);
        loader.addImagesWithLabel("datasets/cifar10-64/class7", 7, 0.3);
        loader.addImagesWithLabel("datasets/cifar10-64/class8", 8, 0.3);
        loader.addImagesWithLabel("datasets/cifar10-64/class9", 9, 0.3);

        Transform transform = new Transform();
        transform.normalizeTanH();

        loader.applyTransform(transform);

        loader.setSeed(42);
        loader.trainTestSplit(0.95);

        loader.batch(64);

        int numClasses = 10;
        int colorChannels = 3;

        Sequential model = new Sequential();

        model.add(Layers.Conv2D(64, colorChannels, 3, "same_padding"));
        model.add(Layers.LeakyReLU(0.1));
        model.add(Layers.MaxPool2D(2, 2));

        model.add(Layers.Conv2D(128, 64, 3, "same_padding"));
        model.add(Layers.LeakyReLU(0.1));
        model.add(Layers.MaxPool2D(2, 2)); 

        model.add(Layers.Dense(16 * 16 * 128, 128));
        model.add(Layers.Dropout(0.3));
        model.add(Layers.LeakyReLU(0.1));

        model.add(Layers.Dense(128, numClasses));
        model.add(Layers.Softmax());

        // model.loadWeights("CNN Cifar64 64.128");


        double oldAccuracy = model.getAccuracy(model.predict(loader.getTestImages()), loader.getTestLabels());


        model.train(loader, 10, 0.01);

        int[] predictions = model.predict(loader.getTestImages());

        JPlot.displayConfusionMatrix(predictions, loader.getTestLabels());

        double newAccuracy = model.getAccuracy(predictions, loader.getTestLabels());
        System.out.println("Test accuracy:" + newAccuracy);

        if (newAccuracy > oldAccuracy) {
            model.saveWeights("CNN Cifar64 64.128");
        }
    }
}