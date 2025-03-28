import java.io.IOException;
import java.util.Random;

import JFlow.JMatrix;
import JFlow.Layers.Layers;
import JFlow.Layers.Sequential;
import JFlow.data.Dataloader;
import JFlow.data.Transform;
import JFlow.utils.JPlot;

public class GANDemo {
    public static void main(String[] args) throws IOException {
        // Build dataset
        Dataloader loader = new Dataloader();
        // loader.loadFromCSV("datasets/MNIST.csv", true, 0.2);
        loader.addImagesWithLabel("datasets/cifar10-32", 0, 0.2);

         // ...

        int imageSize = 32;
        int imageChannels = 3;

        Transform transform = new Transform();
        transform.normalizeTanH();
        // transform.resize(imageSize, imageSize);

        loader.applyTransform(transform);

        loader.setSeed(42);
        
        loader.trainTestSplit(1.0); // All train images

        loader.batch(16);


        // Create Discriminator
        Sequential discriminator = new Sequential();

        discriminator.add(Layers.Conv2D(32, imageChannels, 3, "same_padding")); // Image input
        discriminator.add(Layers.LeakyReLU(0.1));
        discriminator.add(Layers.MaxPool2D(2, 2));

        discriminator.add(Layers.Conv2D(64, 32, 3, "same_padding"));
        discriminator.add(Layers.LeakyReLU(0.1));
        discriminator.add(Layers.MaxPool2D(2, 2));

        // (Flatten implied)
        discriminator.add(Layers.Dense(64 * (imageSize / 4) * (imageSize / 4), 256));
        discriminator.add(Layers.LeakyReLU(0.1));
        discriminator.add(Layers.Dense(256, 1)); 
        discriminator.add(Layers.Sigmoid());


        // Create Generator
        Sequential generator = new Sequential();
        generator.add(Layers.Dense(128, 1024)); // Random noise input

        generator.add(Layers.Reshape(64, 4, 4));

        generator.add(Layers.Upsampling2D(2)); // (64, 8, 8)
        generator.add(Layers.Conv2D(64, 64, 3, "same_padding"));
        generator.add(Layers.LeakyReLU(0.2)); 

        generator.add(Layers.Upsampling2D(2)); // (64, 16, 16)
        generator.add(Layers.Conv2D(32, 64, 3, "same_padding"));
        generator.add(Layers.LeakyReLU(0.2));

        generator.add(Layers.Upsampling2D(2)); // (32, 32, 32)
        generator.add(Layers.Conv2D(imageChannels, 32, 3, "same_padding"));
        generator.add(Layers.Tanh()); // (imageChannels, 32, 32)

        // discriminator.setDebug(true);


        int epochs = 20; // Number of training iterations
        int batchSize = 16; // Batch size for training

        double discLearningRate = 0.0001;
        double genLearningRate = 0.003;


        // Get real images (from dataset)
        double[][] realImages = loader.getBatchesAsArray(); // (Numbatches, batchSize * channels * height * width)
        int numBatches = loader.numBatches();


        double[] displayImage = null;

        // generator.loadWeights("GAN_generator MNIST");
        // discriminator.loadWeights("GAN_discriminator MNIST");

        for (int epoch = 0; epoch < epochs; epoch++) {
            double realLoss = 0.0;
            double fakeLoss = 0.0;
            double generatorLoss = 0.0;
        
            for (int i = 0; i < numBatches; i++) {
                // Generate fake images for the current batch
                JMatrix randomNoise = generateRandomNoise(batchSize, 128); // (batchSize, 128)
                JMatrix generated = generator.forward(randomNoise, true); // (batchSize, imageSize * imageSize * channels)

                displayImage = generated.get(0);

                // JPlot.displayImage(displayImage, 20, imageChannels, "Generated");
        
                // Train Discriminator on real images
                JMatrix realOutput = discriminator.forward(new JMatrix(realImages[i], batchSize, 
                    imageChannels, imageSize, imageSize), true);
                realLoss += binaryCrossEntropyLoss(realOutput, 1, batchSize);
                // for (int b = 0; b < batchSize; b++) {
                //     realLoss += binaryCrossEntropyLoss(realOutput.getMatrix()[b], 1); // Label 1 for real images
                // }
    
                discriminator.backward(realOutput.subtract(1), discLearningRate);
        
                // Train Discriminator on fake images
                JMatrix fakeOutput = discriminator.forward(generated, true);
                fakeLoss += binaryCrossEntropyLoss(fakeOutput, 0, batchSize);
                // for (int b = 0; b < batchSize; b++) {
                //     fakeLoss += binaryCrossEntropyLoss(fakeOutput.getMatrix()[b], 0); // Label 0 for fake images
                // }
        
                // Error
                discriminator.backward(fakeOutput, discLearningRate); // Label 0 for fake  
        
                // Train Generator: Generator tries to fool Discriminator
                JMatrix generatorOutput = discriminator.forward(generated, false);
                // for (int b = 0; b < batchSize; b++) {
                //     generatorLoss += binaryCrossEntropyLoss(generatorOutput, 1, batchSize); // We want the generator to fool the discriminator
                // }
                // generatorLoss += binaryCrossEntropyLoss(generatorOutput, 1, batchSize);
        
                // JMatrix gradD_fake = generatorOutput.subtract(1); 

                generatorLoss += binaryCrossEntropyLoss(generatorOutput, 1, batchSize);
                JMatrix gradD_fake = generatorOutput.subtract(1);
        
                // Backprop through discriminator to get ∂Loss/∂FakeImage
                discriminator.backward(gradD_fake, discLearningRate);
                JMatrix gradG_output = discriminator.getLayerGradient(0);  // Get gradient w.r.t input
        
                // Update generator
                generator.backward(gradG_output, genLearningRate); 

                // try {
                //     Thread.sleep(100000);
                // } catch (Exception e){}
        
                System.out.print("\rBatch " + (i + 1) + " / " + numBatches);


            }
        
            // Average the losses over batches
            realLoss /= numBatches;
            fakeLoss /= numBatches;
            generatorLoss /= numBatches;

            JPlot.displayImage(displayImage, 20, imageChannels, "Generated " + (epoch + 1));

        
            System.out.println("\nEpoch " + (epoch + 1) + ": D_real Loss: " + realLoss + ", D_fake Loss: " + fakeLoss + ", G Loss: " + generatorLoss);
        
        
            generator.saveWeights("GAN_generator Cifar 32 "  + (epoch + 1));
            discriminator.saveWeights("GAN_discriminator Cifar 32 " + (epoch + 1));
        }
    }



    // Binary Cross-Entropy Loss
    public static double binaryCrossEntropyLoss(double predicted, double actual) {
        return -(actual * Math.log(predicted) + (1 - actual) * Math.log(1 - predicted));
    }

    public static double binaryCrossEntropyLoss(JMatrix output, double label, int batchSize) {
        double loss = 0.0;
        double[] outputMatrix = output.getMatrix();
        for (int i = 0; i < output.size(); i++) {
            double y_hat = outputMatrix[i];  // Discriminator output (sigmoid result)
            loss += -label * Math.log(y_hat + 1e-7) - (1 - label) * Math.log(1 - y_hat + 1e-7);
        }
        return loss / batchSize;  // Average over batch
    }
    public static JMatrix generateRandomNoise(int batchSize, int noiseDim) {
        Random random = new Random();
        double[] noise = new double[batchSize * noiseDim];
    
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < noiseDim; j++) {
                noise[i * noiseDim + j] = 2 * random.nextDouble() - 1; // Uniform noise in range [-1,1]
            }
        }
        return new JMatrix(noise, batchSize, noiseDim, 1, 1);
    }
    
}
