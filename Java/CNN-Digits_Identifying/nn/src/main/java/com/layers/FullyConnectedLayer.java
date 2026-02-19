package com.layers;

import java.util.List;
import java.util.Random;

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

        this._weights = new double[_outLength][_outLength];
        setRandomWeights();
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return vector;
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);

        if (_nextLayer != null)
            return _nextLayer.getOutput(input);
        else
            return forwardPass;
    }

    @Override
    public void backPropagation(List<double[][]> dld0) {
        double[] vector = matrixToVector(dld0);
        backPropagation(vector);
    }

    @Override
    public void backPropagation(double[] dld0) {
        double d0dz;
        double dzdw;
        double dldw;
        double dzdx;

        double[] dldx = new double[_inLength];

        for (int k = 0; k < _inLength; k++) {
            double dldx_sum = 0;
            for (int j = 0; j < _outLength; j++) {
                d0dz = derivaticeReLU(lastZ[j]);
                dzdw = lastX[k];
                dzdx = _weights[k][j];

                dldw = dld0[j] * d0dz * dzdw;

                _weights[k][j] -= dldw * _learningRate;

                dldx_sum += dld0[j] * d0dz * dzdx;
            }
            dldx[k] = dldx_sum;
        }
        if (_previousLayer != null) _previousLayer.backPropagation(dldx);
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

    public final void setRandomWeights() {
        Random random = new Random(SEED);

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                _weights[i][j] = random.nextGaussian();
            }
        }
    }

    public double reLU(double input) {
        if (input <= 0)
            return leak;
        else
            return input;
    }

    public double derivaticeReLU(double input) {
        if (input <= 0)
            return leak;
        else
            return 1;
    }

    private double[] fullyConnectedForwardPass(double[] input) {
        lastX = input;

        double[] z = new double[_outLength];
        double[] out = new double[_outLength];

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                z[j] += input[i] * _weights[i][j];
            }
        }

        lastZ = z;

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                out[j] = reLU(z[j]);
            }
        }

        return out;
    }
}
