package com.cnn;

import java.util.List;

import com.data.DataReader;
import com.data.Image;

public class Main {
    public static void main(String[] args) {
        List<Image> images = new DataReader(28, 28).readFromResources("data/mnist_train.csv");
    }
}
