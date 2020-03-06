package org.nd4j.autodiff.nn;

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class NoisyMnistDataSetIterator extends EmnistDataSetIterator {
    private double noiseRatio;

    public NoisyMnistDataSetIterator(double noiseRatio, int batchSize, boolean train, int seed) throws IOException {
        super(Set.BALANCED, batchSize, train, seed);
        this.noiseRatio = noiseRatio;
    }

    @Override
    public DataSet next() {
        DataSet ds = super.next();
        INDArray x = ds.getFeatures();
        INDArray bernoulli = Nd4j.randomBernoulli(noiseRatio, x.shape());
        INDArray rand = Nd4j.rand(x.shape());
        ds.setFeatures(bernoulli.mul(rand).add(x));

        return ds;
    }
}
