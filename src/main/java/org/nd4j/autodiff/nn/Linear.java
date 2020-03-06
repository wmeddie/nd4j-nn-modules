package org.nd4j.autodiff.nn;

import org.nd4j.autodiff.eager.Tensor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.impl.ReluUniformInitScheme;
import org.nd4j.weightinit.impl.UniformInitScheme;

public class Linear extends Module {
    long inFeatures;
    long outFeatures;

    @Param Parameter weight;
    @Param Parameter bias;

    public Linear(long inFeatures, long outFeatures, boolean bias) {
        super();
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.weight = new TensorParameter(outFeatures, inFeatures);
        if (bias) {
            this.bias = new TensorParameter(outFeatures);
        }
        resetParameters();
    }

    public Linear(long inFeatures, long outFeatures) {
        this(inFeatures, outFeatures, true);
    }

    @Override
    protected Tensor forward(Tensor input) {
        if (bias != null) {
            return t(nn.linear(input, input.getSameDiff().transpose(weight), bias));
        } else {
            return t(input.mmul(input.getSameDiff().transpose(weight)));
        }
    }

    @Override
    protected void resetParameters() {
        super.resetParameters();

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            weight.setArray(new ReluUniformInitScheme('c', inFeatures).create(dataType, weight.getShape()));
            if (bias != null) {
                bias.setArray(new UniformInitScheme('c', inFeatures).create(dataType, bias.getShape()));
            }
        }
    }

    @Override
    public String toString() {
        return String.format("inFeatures=%d, outFeatures=%d, bias=%b", inFeatures, outFeatures, bias != null);
    }

}
