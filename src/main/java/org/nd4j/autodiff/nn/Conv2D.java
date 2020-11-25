package org.nd4j.autodiff.nn;

import lombok.ToString;
import org.nd4j.autodiff.eager.Tensor;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.impl.ReluUniformInitScheme;
import org.nd4j.weightinit.impl.UniformInitScheme;

@ToString
public class Conv2D extends Module {
    int inChannels;
    int outChannels;
    Conv2DConfig config;

    @Param Parameter weight;
    @Param Parameter bias;

    public Conv2D(
            int inChannels,
            int outChannels,
            Pair<Integer, Integer> kernelSize,
            Pair<Integer, Integer> stride,
            Pair<Integer, Integer> padding,
            Pair<Integer, Integer> dilation,
            int groups,
            boolean bias,
            PaddingMode paddingMode) {
        super();

        this.inChannels = inChannels;
        this.outChannels = outChannels;

        weight = new TensorParameter(kernelSize.getFirst(), kernelSize.getSecond(), inChannels, outChannels);
        if (bias) {
            this.bias = new TensorParameter(1L, outChannels);
        }

        config = Conv2DConfig.builder()
                .kH(kernelSize.getFirst())
                .kW(kernelSize.getSecond())
                .pH(padding.getFirst())
                .pW(padding.getSecond())
                .sH(stride.getFirst())
                .sW(stride.getSecond())
                .dH(dilation.getFirst())
                .dW(dilation.getSecond())
                .isSameMode(paddingMode == PaddingMode.SAME)
                .dataFormat("NCHW")
                .build();

        resetParameters();
    }

    public Conv2D(int inChannels, int outChannels, int kernelSize) {
        this(inChannels, outChannels, Pair.of(kernelSize, kernelSize), Pair.of(1, 1), Pair.of(0, 0), Pair.of(1, 1), 1, true, PaddingMode.VALID);
    }

    public Conv2D(int inChannels, int outChannels, Pair<Integer, Integer> kernelSize) {
        this(inChannels, outChannels, kernelSize, Pair.of(1, 1), Pair.of(0, 0), Pair.of(1, 1), 1, true, PaddingMode.VALID);
    }


    public Conv2D(int inChannels, int outChannels, Pair<Integer, Integer> kernelSize, Pair<Integer, Integer> stride) {
        this(inChannels, outChannels, kernelSize, stride, Pair.of(0, 0), Pair.of(1, 1), 1, true, PaddingMode.VALID);
    }

    public Conv2D(int inChannels, int outChannels, Pair<Integer, Integer> kernelSize, Pair<Integer, Integer> stride, Pair<Integer, Integer> padding) {
        this(inChannels, outChannels, kernelSize, stride, padding, Pair.of(1, 1), 1, true, PaddingMode.VALID);
    }

    public Conv2D(int inChannels, int outChannels, Pair<Integer, Integer> kernelSize, Pair<Integer, Integer> stride, Pair<Integer, Integer> padding, Pair<Integer, Integer> dialation) {
        this(inChannels, outChannels, kernelSize, stride, padding, dialation, 1, true, PaddingMode.VALID);
    }


    @Override
    protected Tensor forward(Tensor input) {
        if (bias != null) {
            return t(cnn.conv2d(input, weight, bias, config));
        } else {
            return t(cnn.conv2d(input, weight, config));
        }
    }

    @Override
    protected void resetParameters() {
        super.resetParameters();

        try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            weight.setArray(new ReluUniformInitScheme('c', inChannels).create(dataType, weight.getShape()));
            bias.setArray(new UniformInitScheme('c', inChannels).create(dataType, bias.getShape()));
        }
    }
}
