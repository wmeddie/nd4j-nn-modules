package org.nd4j.autodiff.nn;

import org.nd4j.autodiff.eager.Tensor;
import org.nd4j.autodiff.samediff.SDVariable;

public class CrossEntropyLoss extends WeightedLoss {

    @Mod final Module model;

    public CrossEntropyLoss(Module model, Parameter weight) {
        super(weight);
        this.model = model;
    }

    public CrossEntropyLoss(Module model) {
        this(model, null);
    }

    @Override
    protected Tensor forward(Tensor modelInput, Tensor target) {
        SDVariable output = model.forward(modelInput);

        if (weight == null) {
            SDVariable losses = loss.softmaxCrossEntropy(null, target, output);
            losses.markAsLoss();
            return t(losses);
        } else {
            SDVariable variable = loss.weightedCrossEntropyWithLogits(target, output, weight);
            variable.markAsLoss();
            return t(variable);
        }
    }

    /*@Override
    protected void setTape(SameDiff sd) {
        super.setTape(model.getTape());
    }
     */
}
