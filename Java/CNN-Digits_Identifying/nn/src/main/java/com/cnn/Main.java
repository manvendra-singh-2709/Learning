package com.cnn;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.URISyntaxException;
import static java.util.Collections.shuffle;
import java.util.List;

import com.data.DataReader;
import com.data.Image;
import com.network.NetworkBuilder;
import com.network.NeuralNetwork;

public class Main {
    public static void main(String[] args) {
        boolean enableLogging = false;
        boolean trainModel = false;
        int input = 28;

        long SEED = 42;
        double learningRate = 0.1;
        int epochs = 20;

        if (enableLogging) {
            initLogging(epochs);
        }

        File modelFile = getModelFile(epochs);

        System.out.println("Starting data loading...\n");
        List<Image> imagesTrain = new DataReader(input, input).readFromResources("data/mnist_train.csv");
        List<Image> imagesTest = new DataReader(input, input).readFromResources("data/mnist_test.csv");

        NeuralNetwork net;

        if (!trainModel && modelFile.exists()) {
            System.out.println("Loading model from: " + modelFile.getAbsolutePath());
            net = loadModel(modelFile);
        } else {
            System.out.println("Initializing new network...");
            NetworkBuilder builder = new NetworkBuilder(input, input, 256 * 100);
            builder.addConvolutionLayer(8, 5, 1, learningRate, SEED);
            builder.addMaxPoolLayer(3, 2);
            builder.addFullyConnectedLayer(10, learningRate, SEED);
            net = builder.build();

            if (trainModel) {
                System.out.println("Starting training...");
                for (int i = 0; i < epochs; i++) {
                    shuffle(imagesTrain);
                    long start = System.currentTimeMillis();
                    net.train(imagesTrain);
                    long end = System.currentTimeMillis();
                    float rate = net.test(imagesTest);
                    System.out.println(
                            "Epoch: " + (i + 1) + " | Success Rate: " + rate + " | Time: " + (end - start) + "ms");
                }
                saveModel(net, modelFile);
            }
        }

        float finalRate = net.test(imagesTest);
        System.out.println("Final success rate: " + finalRate);
    }

    private static File getModelFile(int epochs) {
        try {
            File classFile = new File(Main.class.getProtectionDomain().getCodeSource().getLocation().toURI());
            File projectRoot = classFile.getParentFile().getParentFile();
            File modelDir = new File(projectRoot, "src/main/java/com/model");
            if (!modelDir.exists()) {
                modelDir.mkdirs();
            }
            return new File(modelDir, "model_epoch_" + epochs + ".bin");
        } catch (URISyntaxException e) {
            return new File("trained_model.bin");
        }
    }

    private static void saveModel(NeuralNetwork net, File file) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
            oos.writeObject(net);
            System.out.println("Model saved successfully.");
        } catch (Exception e) {
            System.err.println("Failed to save model: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static NeuralNetwork loadModel(File file) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            return (NeuralNetwork) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Failed to load model: " + e.getMessage());
            return null;
        }
    }

    private static void initLogging(int epochs) {
        try {
            File classFile = new File(Main.class.getProtectionDomain().getCodeSource().getLocation().toURI());
            File projectRoot = classFile.getParentFile().getParentFile();
            File logDir = new File(projectRoot, "src/main/java/com/logs");

            if (!logDir.exists()) {
                logDir.mkdirs();
            }

            File logFile = new File(logDir, "training_epoch_" + epochs + ".log");
            final FileOutputStream fos = new FileOutputStream(logFile);
            final PrintStream consoleOut = System.out;

            PrintStream multiOut = new PrintStream(new OutputStream() {
                @Override
                public void write(int b) throws IOException {
                    consoleOut.write(b);
                    fos.write(b);
                }

                @Override
                public void flush() throws IOException {
                    consoleOut.flush();
                    fos.flush();
                }
            });

            System.setOut(multiOut);
            System.setErr(multiOut);
            System.out.println("Logging initialized at: " + logFile.getAbsolutePath());
        } catch (URISyntaxException | IOException e) {
            System.err.println("Logging setup failed: " + e.getMessage());
        }
    }
}