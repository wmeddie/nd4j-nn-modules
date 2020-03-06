package org.nd4j.autodiff.nn;

import org.nd4j.autodiff.samediff.SDVariable;

abstract class WeightedLoss extends Module {
    @Buff Parameter weight;

    WeightedLoss(Parameter weight) {
        this.weight = weight;
    }
}
