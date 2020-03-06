package org.nd4j.autodiff.nn;

public class TensorParameter extends Parameter {
    long[] shape;

    public TensorParameter(long... shapes) {
        super(null);
        shape = shapes;
    }

    @Override
    public long[] getShape() {
        return shape;
    }
}
