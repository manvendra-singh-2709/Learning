package com.layers;

import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

@SuppressWarnings("all")
public class FullyConnectedLayer extends Layer {

    private final double leak = 0.000001;
    private final double[][] _weights;
    private final int _inLength;
    private final int _outLength;
    private final double _learningRate;
    private final long SEED;

    private double[] lastZ;
    private double[] lastX;

    public FullyConnectedLayer(int _inLength, int _outLength, long SEED, double _learningRate) {
        this._inLength = _inLength;
        this._outLength = _outLength;
        this.SEED = SEED;
        this._learningRate = _learningRate;

        this._weights = new double[_inLength][_outLength];
        setRandomWeights();
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        return getOutput(matrixToVector(input));
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);
        return (_nextLayer != null) ? _nextLayer.getOutput(forwardPass) : forwardPass;
    }

    @Override
    public void backPropagation(List<double[][]> dld0) {
        backPropagation(matrixToVector(dld0));
    }

    @Override
    public void backPropagation(double[] dld0) {
        double[] dldx = new double[_inLength];
        double[] preComputedDeriv = new double[_outLength];

        for (int j = 0; j < _outLength; j++) {
            preComputedDeriv[j] = dld0[j] * derivaticeReLU(lastZ[j]);
        }

        IntStream.range(0, _inLength).parallel().forEach(k -> {
            double dldx_sum = 0;
            double lastXk = lastX[k];
            double[] weightRow = _weights[k];

            for (int j = 0; j < _outLength; j++) {
                double gradientBase = preComputedDeriv[j];

                dldx_sum += gradientBase * weightRow[j];
                weightRow[j] -= gradientBase * lastXk * _learningRate;
            }
            dldx[k] = dldx_sum;
        });

        if (_previousLayer != null) {
            _previousLayer.backPropagation(dldx);
        }
    }

    private double[] fullyConnectedForwardPass(double[] input) {
        lastX = input;
        double[] z = new double[_outLength];
        double[] out = new double[_outLength];

        for (int i = 0; i < _inLength; i++) {
            double inputVal = input[i];
            double[] weightRow = _weights[i];
            for (int j = 0; j < _outLength; j++) {
                z[j] += inputVal * weightRow[j];
            }
        }

        lastZ = z;
        for (int j = 0; j < _outLength; j++) {
            out[j] = reLU(z[j]);
        }

        return out;
    }

    public final void setRandomWeights() {
        Random random = new Random(SEED);
        double scale = Math.sqrt(2.0 / _inLength);
        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                _weights[i][j] = random.nextGaussian() * scale;
            }
        }
    }

    public double reLU(double input) {
        return input <= 0 ? leak : input;
    }

    public double derivaticeReLU(double input) {
        return input <= 0 ? leak : 1;
    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputColumns() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return _outLength;
    }
}