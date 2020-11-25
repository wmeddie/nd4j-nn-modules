package org.nd4j.autodiff.nn;

import lombok.ToString;
import org.nd4j.autodiff.eager.Tensor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;

@ToString
public class MaxPool2D extends Module {
    private final Pooling2DConfig config;

    public MaxPool2D(Pair<Integer, Integer> kernelSize, Pair<Integer, Integer> stride) {
        config = Pooling2DConfig.builder()
                .kH(kernelSize.getFirst())
                .kW(kernelSize.getSecond())
                .dH(1)
                .dW(1)
                .isSameMode(false)
                .pH(0)
                .pW(0)
                .sH(kernelSize.getFirst())
                .sW(kernelSize.getSecond())
                .isNHWC(false)
                .build();
    }

    @Override
    protected Tensor forward(Tensor in) {
        return t(cnn.maxPooling2d(in, config));
    }
}
