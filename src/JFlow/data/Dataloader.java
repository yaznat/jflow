package JFlow.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

public class Dataloader {
    private ArrayList<Image> images = new ArrayList<Image>();

    private Random random = new Random(0);

    private ArrayList<Image> trainImages = new ArrayList<Image>(); 
    private ArrayList<Image> testImages = new ArrayList<Image>(); 

    private int batchSize = -1;
    private int numBatches;

    public Dataloader () {}

    // Apply a transform to all of the images
    public void applyTransform(Transform transform) {
        for (Image image : images) {
            for (Function<double[][][], double[][][]> function : transform.getTransforms()) {
                image.applyTransform(function);
            }
        }
    }

    // load a folder of images into the dataset with a certain label
    public void addImagesWithLabel(String directory, int label, double percentOfDirectory) {
        File dir = new File(directory);

        File[] files = dir.listFiles();

        int numImages = (int)(files.length * percentOfDirectory);

        for (int i = 0; i < numImages; i++) {
            if (files[i].getAbsolutePath().endsWith(".png") || 
                    files[i].getAbsolutePath().endsWith(".png"))
                images.add(new Image(files[i].getAbsolutePath(), label));
        }
    }

    // load flattened images from csv
    public void loadFromCSV(String path, boolean areLabelsFirstItem, double percent) {
        ArrayList<Image> loadedImages = new ArrayList<Image>();
        try(BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line; String[] split;
            int label = 0;
            while((line = br.readLine()) != null){
                split = line.split(",");
                double[] image;
                int index = 0;
                if (areLabelsFirstItem) {
                    label = Integer.valueOf(split[0]);
                    index = 1;
                    image = new double[split.length - 1];
                } else {
                    image = new double[split.length];
                }
                for (int i = index; i < split.length; i++) {
                    image[i - index] = Double.parseDouble(split[i]);
                }
                
                loadedImages.add(new Image(image, label));
            }
        } catch (Exception e) {
            System.err.println(e);
        }
        int imagesToKeep = (int)(percent * loadedImages.size());

        for (int i = 0; i< imagesToKeep; i++) {
            images.add(loadedImages.get(i));
        }
    }
    public void addArrayAsImage(double[][][] array, int label) {
        images.add(new Image(array, label));
    }
    public void clear() {
        images = new ArrayList<Image>();
        trainImages = new ArrayList<Image>();
        testImages = new ArrayList<Image>();
    }

    public void setSeed(long seed) {
        random.setSeed(seed);
    }

    // Set batch size
    public void batch(int batchSize) {
        this.batchSize = batchSize;
        this.numBatches = trainImages.size()/batchSize;
    }

    public void shuffle() {
        for (int i = 0; i < images.size(); i++) {
            int randIndex = random.nextInt(images.size());

            Image temp = images.get(i);
            images.set(i, images.get(randIndex));

            images.set(randIndex, temp);
        }
    }

    public Image get(int index) {
        return images.get(index);
    }

    public int size() {
        return images.size();
    }

    private int numberOfBatches() {
        if (trainImages.isEmpty()) {
            this.numBatches = images.size() / batchSize;
            return images.size() / batchSize;
        }
        this.numBatches = trainImages.size() / batchSize;
        return trainImages.size() / batchSize;
    }

    public int numBatches() {
        return numBatches;
    }

    public List<Image> getBatch(int index) {
        int beginIndex = batchSize * index;
        int endIndex = beginIndex + batchSize;
        if (endIndex < images.size()) {
            return images.subList(beginIndex, endIndex);
        }
        return null;
    }

    // Get batches for training
    public List<List<Image>> getBatches() {
        if (trainImages.isEmpty()) {
            trainImages = images;
        }
        List<List<Image>> batches = new ArrayList<>();
        for (int i = 0; i < numberOfBatches(); i++) {
            int beginIndex = batchSize * i;
            int endIndex = beginIndex + batchSize;
            batches.add(trainImages.subList(beginIndex, endIndex));
        }
        return batches;
    }

    // Return train images batched in a 2D array
    // public double[][] getBatchesAsArray() {
    //     List<List<Image>> batches = getBatches();
    //     int numBatches = numBatches();
    //     int batchSize = batches.get(0).size();
    //     int imageSize = batches.get(0).get(0).getHeight() * 
    //     batches.get(0).get(0).getWidth() * batches.get(0).get(0).numChannels();
    //     double [][] batchArray = new double[numBatches][batchSize * imageSize];

    //     boolean inconsistent = false;
    //     for (int i = 0; i < numBatches; i++) {
    //         double[] flatBatch = new double[batchSize * imageSize];
    //         for (int j = 0; j < batchSize; j++) {
    //             double[] flat = batches.get(i).get(j).getFlat();
    //             try {
    //                 System.arraycopy(flat, 0, flatBatch, j * imageSize, imageSize);
    //             } catch (ArrayIndexOutOfBoundsException e) {
    //                 // If data is not all of the same type, remove the batch for consistency
    //                 batches.remove(i);
    //                 i++;
    //                 inconsistent = true;
    //             }
                
    //         }   
    //         if (!inconsistent) {
    //             batchArray[i] = flatBatch;
    //         }
            
    //     }
    //     return batchArray;
    // }
    public double[][] getBatchesAsArray() {
        List<List<Image>> batches = getBatches();
        int numBatches = getBatches().size();
        int batchSize = batches.get(0).size();
        int imageSize = batches.get(0).get(0).getHeight() * 
                        batches.get(0).get(0).getWidth() * 
                        batches.get(0).get(0).numChannels();
    
        List<double[]> validBatches = new ArrayList<>();
    
        for (List<Image> batch : batches) {
            double[] flatBatch = new double[batchSize * imageSize];
            boolean valid = true;
    
            for (int j = 0; j < batchSize; j++) {
                double[] flat = batch.get(j).getFlat();
    
                if (flat.length != imageSize) {
                    valid = false;
                    numBatches--;
                    break;  // Stop processing this batch
                }
    
                System.arraycopy(flat, 0, flatBatch, j * imageSize, imageSize);
            }
    
            if (valid) {
                validBatches.add(flatBatch);
            }
        }
        this.numBatches = numBatches;
    
        // Convert List to double[][]
        return validBatches.toArray(new double[0][]);
    }
    // Split images into train and test
    public void trainTestSplit(double percentTrain) {
        Collections.shuffle(images, random);
    
        int numTrainImages = (int) (images.size() * percentTrain);
    
        trainImages = new ArrayList<>(images.subList(0, numTrainImages));
        testImages = new ArrayList<>(images.subList(numTrainImages, images.size()));
    
        System.out.println("Train images: " + trainImages.size());
        System.out.println("Test images: " + testImages.size());
    }

    /*
     * For each image in the dataset add a duplicate,
     * and apply a transform it.
     */
    public void doubleDatasetWithAugmentation(Transform transform) {
        ArrayList<Image> arrayToUse;
        if (trainImages.isEmpty()) {
            arrayToUse = images;
        } else {
            arrayToUse = trainImages;
        }
        int numImages = arrayToUse.size();
        for (int i = 0; i < numImages; i++) {
            Image original = arrayToUse.get(i);
            Image augmented = new Image(original.getData(), original.getLabel());
            for  (Function<double[][][], double[][][]> 
                function : transform.getTransforms()) {
                
                augmented.applyTransform(function);
            }
            arrayToUse.add(augmented);
        }
    }

    public double[][] getTestImagesFlat() {
        if (testImages.isEmpty()) {
            throw new NullPointerException("Test dataset never set");
        }
        int numImages = testImages.size();
        int imageSize = testImages.get(0).getHeight() * testImages.get(0).getWidth();
        double[][] images = new double[numImages][imageSize];

        for (int i = 0; i < numImages; i++) {
            images[i] = testImages.get(i).getFlat();
        }

        return images;
    }

    public double[][][][] getTestImages() {
        if (testImages.isEmpty()) {
            throw new NullPointerException("Test dataset never set");
        }
        int numImages = testImages.size();
        int channels =(testImages.get(0)).numChannels();
        int imageHeight = testImages.get(0).getHeight();
        int imageWidth = testImages.get(0).getWidth();
        double[][][][] images = new double[numImages][channels][imageHeight][imageWidth];

        for (int i = 0; i < numImages; i++) {
            images[i] = testImages.get(i).getData();
        }

        return images;
    }

    public int[] getTestLabels() {
        if (testImages.isEmpty()) {
            throw new NullPointerException("Test dataset never set");
        }
        int numImages = testImages.size();
        int[] labels = new int[numImages];

        for (int i = 0; i < numImages; i++) {
            labels[i] = testImages.get(i).getLabel();
        }

        return labels;
    }
}
