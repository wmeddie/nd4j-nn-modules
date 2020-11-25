package org.nd4j.autodiff.optim;

import org.nd4j.autodiff.nn.Parameter;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.common.function.Supplier;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

public class Sgd extends Optimizer {
    private final double learningRate;
    private final double momentum;
    private final double dampening;
    private final double weightDecay;
    private final boolean nesterov;
    private Map<String, GradientUpdater> updaters;
    private int iteration;


    public Sgd(Collection<Parameter> parameters, double learningRate, double momentum, double dampening, double weightDecay, boolean nesterov) {
        super(parameters);
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.dampening = dampening;
        this.weightDecay = weightDecay;
        this.nesterov = nesterov;

        iteration = 0;
    }

    public Sgd(Collection<Parameter> parameters, double learningRate) {
        this(parameters, learningRate, 0.0, 0.0, 0.0, false);
    }

    @Override
    public double step(Supplier<Double> lossClosure) {
        if (updaters == null) {
            updaters = new HashMap<>();

            for (Parameter param : parameters) {
                if (!nesterov) {
                    updaters.put(param.name(), new org.nd4j.linalg.learning.config.Sgd(0.001).instantiate((INDArray) null, false));
                } else {
                    Nesterovs nesterovs = new Nesterovs(learningRate, momentum);
                    long stateSize = nesterovs.stateSize(param.getArr().length());
                    INDArray view = Nd4j.createUninitialized(1, stateSize);
                    updaters.put(param.name(), nesterovs.instantiate(view, true));
                }
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

            updaters.get(param.name()).applyUpdater(grad.getArr(), iteration, 0);
            param.getArr().subi(grad.getArr());
        }

        iteration++;

        return loss;
    }
}
