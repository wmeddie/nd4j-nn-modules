package org.nd4j.autodiff.optim;

import org.nd4j.autodiff.nn.Parameter;
import org.nd4j.common.function.Supplier;

import java.util.Collection;

public abstract class Optimizer {
    protected Collection<Parameter> parameters;

    public Optimizer(Collection<Parameter> parameters) {
        this.parameters = parameters;
    }

    public void zeroGrad() {
        for (Parameter p : parameters) {
            if (p.hasGradient()) {
                p.gradient().getArr().muli(0.0);
            }
        }
    }

    public abstract double step(Supplier<Double> lossClosure);

    public double step() {
        return step(null);
    }
}
