package org.nd4j.autodiff.optim;

import org.nd4j.autodiff.nn.Parameter;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Supplier;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.Adam;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

public class AdamOptimizer extends Optimizer {
    private Map<String, GradientUpdater> updaters;
    private double learningRate;
    private int iteration;

    public AdamOptimizer(Collection<Parameter> parameters, double learningRate) {
        super(parameters);
        this.learningRate = learningRate;
        this.iteration = 0;
    }

    @Override
    public double step(Supplier<Double> lossClosure) {
        if (updaters == null) {
            updaters = new HashMap<>();

            for (Parameter param : parameters) {
                Adam adam = new Adam(learningRate);
                long stateSize = adam.stateSize(param.getArr().length());
                INDArray view = Nd4j.createUninitialized(1, stateSize);
                updaters.put(param.name(), adam.instantiate(view, true));
            }
        }

        double loss = 0.0;
        if (lossClosure != null) {
            loss = lossClosure.get();
        }

        for (Parameter param : parameters) {
            SDVariable grad = param.gradient();
            if (grad == null) {
                continue;
            }

            updaters.get(param.name()).applyUpdater(grad.getArr().reshape(1, prod(grad.getArr().shape())), iteration, 0);
            param.getArr().subi(grad.getArr());
        }

        iteration++;

        return loss;
    }

    private long prod(long[] shapes) {
        long ret = shapes[0];
        for (int i = 1; i < shapes.length; i++) {
            ret *= shapes[i];
        }
        return ret;
    }
}
