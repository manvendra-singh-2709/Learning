package com.cnn;

import static java.util.Collections.shuffle;
import java.util.List;

import com.data.DataReader;
import com.data.Image;
import com.network.NetworkBuilder;
import com.network.NeuralNetwork;

public class Main {
    public static void main(String[] args) {

        long SEED = 42;
        double leariningRate = 0.1;
        int epochs = 10;

        System.out.println("Starting data loading...\n");
        List<Image> imagesTrain = new DataReader(28, 28).readFromResources("data/mnist_train.csv");
        List<Image> imagesTest = new DataReader(28, 28).readFromResources("data/mnist_test.csv");

        System.out.println("Images Train Size:" + imagesTrain.size());
        System.out.println("Images Test Size :" + imagesTest.size());

        NetworkBuilder builder = new NetworkBuilder(28, 28, 256 * 100);
        builder.addConvolutionLayer(8, 5, 1, leariningRate, SEED);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(10, leariningRate, SEED);

        NeuralNetwork net = builder.build();

        float rate = net.test(imagesTest);
        System.out.println("Pre-trained success rate: " + rate);

        for (int i = 0; i < epochs; i++) {
            shuffle(imagesTrain);
            net.train(imagesTest);
            rate = net.test(imagesTest);
            System.out.println("Epoch: " + (i+1) + ", Success Rate: " + rate);
        }

        shuffle(imagesTrain);
        net.train(imagesTest);
        rate = net.test(imagesTest);
        System.out.println("Pre-trained succes rate: " + rate);
    }
}
