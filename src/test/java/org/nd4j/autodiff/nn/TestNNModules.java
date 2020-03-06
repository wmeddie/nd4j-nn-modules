package org.nd4j.autodiff.nn;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import lombok.var;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.autodiff.eager.Tensor;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.optim.AdamOptimizer;
import org.nd4j.autodiff.optim.Optimizer;
import org.nd4j.autodiff.optim.Sgd;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.guava.base.Stopwatch;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;

@Slf4j
public class TestNNModules {
    @BeforeClass
    public static void before() {
        Nd4j.getRandom().setSeed(42);
    }

    private static class MyModel extends Module {
        @Mod Linear fc1;
        @Mod Linear fc2;

        MyModel() {
            super();
            fc1 = new Linear(10, 5, true);
            fc2 = new Linear(5, 2, true);
        }

        @Override
        protected Tensor forward(Tensor x) {
            x = t(nn.relu(fc1.forward(x), 0.0));
            x = fc2.forward(x);

            return x;
        }
    }

    @Test
    public void testForward() {
        MyModel model = new MyModel();

        INDArray input = Nd4j.ones(1, 10);
        INDArray output = model.call(input);

        assertNotNull(output);
    }

    @Test
    public void testLoss() {
        MyModel model = new MyModel();

        Module criterion = new CrossEntropyLoss(model);
        Optimizer optimizer = new Sgd(model.parameters(), 0.1);

        INDArray inputs = Nd4j.rand(1, 10);
        INDArray labels = Nd4j.rand(1, 2);


        optimizer.zeroGrad();

        INDArray outputs = model.call(inputs);
        INDArray loss = criterion.call(inputs, labels);
        criterion.backward();

        optimizer.step();

        INDArray afterOutput = model.call(inputs);

        assertNotEquals(outputs, afterOutput);

        optimizer.zeroGrad();
        outputs = model.call(inputs);
        loss = criterion.call(inputs, labels);
        criterion.backward();

        optimizer.step();

        INDArray afterTwoSteps = model.call(inputs);

        assertNotEquals(afterOutput, afterTwoSteps);
    }

    @Test
    public void testTrainingLoop() {
        MyModel model = new MyModel();

        Module criterion = new CrossEntropyLoss(model);
        Optimizer optimizer = new Sgd(model.parameters(), 0.1);

        INDArray inputs = Nd4j.repeat(Nd4j.linspace(0, 1, 10), 10).reshape(10, 10);
        INDArray labels = Nd4j.repeat(Nd4j.linspace(0, 1, 2), 10).reshape(10, 2);

        INDArray beforeOutput = model.call(inputs);
        for (int i = 0; i < 100; i++) {
            optimizer.zeroGrad();
            criterion.call(inputs, labels);
            criterion.backward();
            optimizer.step();
        }
        INDArray afterOutput = model.call(inputs);

        assertNotEquals(beforeOutput, afterOutput);
    }

    static class MnistModel extends Module {
        @Mod Conv2D conv1;
        @Mod Conv2D iconv1;
        @Mod Conv2D conv2;
        @Mod Conv2D iconv2;
        @Mod MaxPool2D pool;
        @Mod Dropout2D dropout;
        @Mod Linear fc1;

        public MnistModel(int out) {
            super();

            pool = new MaxPool2D(Pair.of(2, 2), Pair.of(2, 2));
            dropout = new Dropout2D(0.8);
            conv1 = new Conv2D(1, 6, 5);
            iconv1 = new Conv2D(1, 6, 5);
            conv2 = new Conv2D(6, 16, 5);
            iconv2 = new Conv2D(6, 16, 5);
            fc1 = new Linear(16 * 4 * 4, out);
            //fc1 = new Linear(6 * 12 * 12, 10);
        }

        private Tensor relu(Tensor x) {
            return t(nn.relu(x, 0.0));
        }

        private Tensor irelu(Tensor x) {
            return t(nn.relu(x, 0.0).mul(-1.0));
        }

        @Override
        protected Tensor forward(Tensor x) {
            x = t(x.reshape(-1, 1, 28, 28));
            var c1 = relu(conv1.forward(x));
            var c2 = irelu(iconv1.forward(x));
            x = t(c1.add(c2));
            x = pool.forward(x);

            c1 = relu(dropout.forward(conv2.forward(x)));
            c2 = irelu(dropout.forward(iconv2.forward(x)));
            x = t(c1.add(c2));
            x = pool.forward(x);

            x = t(x.reshape(-1, 16 * 4 * 4));
            x = fc1.forward(x);

            return x;
        }
    }

    @Test
    public void testMnist() throws IOException {
        val trainData = new NoisyMnistDataSetIterator(0.5, 64, true, 42);
        val testData = new NoisyMnistDataSetIterator(0.5, 64, false, 42);

        val model = new MnistModel(trainData.getLabels().size());
        val criterion = new CrossEntropyLoss(model);
        //val optimizer = new Sgd(model.parameters(), 0.001, 0.9, 0.0, 0.0, false);
        val optimizer = new AdamOptimizer(model.parameters(), 0.01);

        val stopwatch = Stopwatch.createStarted();
        var i = 0;

        for (int epoch = 1; epoch <= 10; epoch++) {
            trainData.reset();

            while (trainData.hasNext()) {
                val dataSet = trainData.next();

                optimizer.zeroGrad();
                val loss = criterion.call(dataSet.getFeatures(), dataSet.getLabels());
                criterion.backward();
                optimizer.step();
                i++;

                if (i % 100 == 99) {
                    val lossValue = loss.getDouble(0);
                    System.out.println("Iterations: " + (i + 1) + " Loss: " + lossValue);
                }
            }
            System.out.println("Finished epoch: " + epoch);
        }
        val elapsed = stopwatch.elapsed(TimeUnit.SECONDS);
        System.out.println("Training Time: " + elapsed + "s");

        val evaluation = new Evaluation(trainData.getLabels().size());
        while (testData.hasNext()) {
            val dataSet = testData.next();
            val predictions = model.call(dataSet.getFeatures());
            evaluation.eval(dataSet.getLabels(), predictions);
        }

        System.out.println(evaluation);
    }

    @Test
    public void testGraphPerformance() throws IOException {
        val trainData = new MnistDataSetIterator(128, true, 42);
        val testData = new MnistDataSetIterator(128, false, 42);

        val model = new MnistModel(10);
        val criterion = new CrossEntropyLoss(model);

        val data = trainData.next();
        criterion.apply(data.getFeatures(), data.getLabels());
        trainData.reset();

        val sd = criterion.getTape();
        val optimizer = new org.nd4j.linalg.learning.config.Sgd(0.001);

        sd.setTrainingConfig(new TrainingConfig.Builder()
                .l2(1e-4)
                .updater(optimizer)
                .dataSetFeatureMapping("input0")
                .dataSetLabelMapping("input1")
                .build()
        );
        sd.setListeners(new ScoreListener(100));

        val started = Stopwatch.createStarted();
        sd.fit(trainData, 3);
        val elapsed = started.elapsed(TimeUnit.SECONDS);
        System.out.println("Training time: " + elapsed + "s");

        val evaluation = new Evaluation(10);
        while (testData.hasNext()) {
            val dataSet = testData.next();
            model.eval();
            val predictions = model.call(dataSet.getFeatures());
            evaluation.eval(dataSet.getLabels(), predictions);
        }

        System.out.println(evaluation);
    }
}
