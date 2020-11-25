package org.nd4j.autodiff.nn;

import org.nd4j.autodiff.eager.Tensor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.buffer.DataType;

public class Dropout2D extends Module {
    private final double keepProbability;

    public Dropout2D() {
        this(0.5);
    }

    public Dropout2D(double keepProbability) {
        //basically bernoulli random, divide by p, multiply by activations
        this.keepProbability = keepProbability;
    }

    @Override
    protected Tensor forward(Tensor in) {
        if (getTrain()) {
            SDVariable bernoulli = random.bernoulli(keepProbability, DataType.FLOAT, in.getShape());

            return t(in.mul(bernoulli));
        } else {
            return in;
        }
    }
}
