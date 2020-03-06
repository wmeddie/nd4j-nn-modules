package org.nd4j.autodiff.nn;

import org.nd4j.autodiff.eager.Tensor;
import org.nd4j.autodiff.samediff.SDVariable;

/**
 * A placeholder identity operator that is argument-insensitive.
 */
public class Identity extends Module {

    public Identity() {
        super();
    }

    @Override
    protected Tensor forward(Tensor in) {
        return in;
    }
}
