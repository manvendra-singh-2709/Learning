package com.network;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import com.data.Image;
import static com.data.MatrixUtility.add;
import static com.data.MatrixUtility.multiply;
import com.layers.Layer;

public class NeuralNetwork implements Serializable {

    private static final long serialVersionUID = 1L;

    List<Layer> _layers;
    double scaleFactor;

    public NeuralNetwork(List<Layer> _layers, double scaleFactor) {
        this._layers = _layers;
        this.scaleFactor = scaleFactor;

        linkLayers();
    }

    private void linkLayers() {
        if (_layers.size() <= 1)
            return;

        for (int i = 0; i < _layers.size(); i++) {
            if (i == 0) {
                _layers.get(i).setNextLayer(_layers.get(i + 1));
            } else if (i == _layers.size() - 1) {
                _layers.get(i).setPreviousLayer(_layers.get(i - 1));
            } else {
                _layers.get(i).setPreviousLayer(_layers.get(i - 1));
                _layers.get(i).setNextLayer(_layers.get(i + 1));
            }
        }
    }

    public double[] getErrors(double[] networkOutput, int correctAnswer) {
        int numClasses = networkOutput.length;

        double[] expected = new double[numClasses];

        expected[correctAnswer] = 1;

        return add(networkOutput, multiply(expected, -1));
    }

    private int getMaxIndex(double[] in) {

        double max = 0;
        int index = 0;

        for (int i = 0; i < in.length; i++) {
            if (in[i] >= max) {
                max = in[i];
                index = i;
            }
        }
        return index;
    }

    public int guess(Image image) {
        List<double[][]> inList = new ArrayList<>();
        inList.add(multiply(image.getData(), 1.0 / scaleFactor));

        double[] out = _layers.get(0).getOutput(inList);
        int guess = getMaxIndex(out);

        return guess;
    }

    public float test(List<Image> images) {
        int correct = 0;

        for (Image img : images) {
            int guess = guess(img);

            if (guess == img.getLabel()) {
                correct++;
            }
        }

        return (float) correct / images.size();
    }

    public void train(List<Image> images) {
        for (Image img : images) {
            List<double[][]> inList = new ArrayList<>();
            inList.add(multiply(img.getData(), 1.0 / scaleFactor));

            double[] out = _layers.get(0).getOutput(inList);

            double[] dld0 = getErrors(out, img.getLabel());

            _layers.get(_layers.size() - 1).backPropagation(dld0);
        }
    }

    public void saveModel(String fileName) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName))) {
            out.writeObject(this);
            System.out.println("Model saved to " + fileName);
        } catch (IOException e) {
        }
    }

    public static NeuralNetwork loadModel(String fileName) {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName))) {
            return (NeuralNetwork) in.readObject();
        } catch (IOException | ClassNotFoundException e) {
            return null;
        }
    }
}