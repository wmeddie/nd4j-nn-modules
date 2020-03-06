package org.nd4j.autodiff.nn;

import org.nd4j.autodiff.eager.Tensor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.ops.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Consumer;
import org.nd4j.linalg.primitives.Pair;

import java.lang.annotation.Annotation;
import java.lang.reflect.Field;
import java.util.*;


/**
 * Base class for autodiff neural network models.
 *
 * You can create models by subclassing Module adding instance variables that
 * are Parameter subclasses and implementing the forward pass function.
 *
 * A simple example model is shown below:
 *
 * public class MyModel extends Module {
 *    \@Mod Conv2d conv1;
 *    \@Mod Conv2d conv2;
 *
 *    public MyModel() {
 *        conv1 = nn.conv2d(1, 20, 5);
 *        conv2 = nn.conv2d(20, 20, 5);
 *    }
 *
 *    \@Override SDVariable forward(SDvariable x) {
 *        x = f.relu(conv1(x))
 *        return f.relu(conv2(x))
 *    }
 * }
 */
public abstract class Module {
    protected DataType dataType;
    protected SDNN nn;
    protected SDCNN cnn;
    protected SDRNN rnn;
    protected SDLoss loss;
    protected SDImage image;
    protected SDMath math;
    protected SDBitwise bitwise;
    protected SDRandom random;

    private boolean training = true;
    private boolean dynamic = true;
    private LinkedHashMap<String, Field> parameters;
    private LinkedHashMap<String, Field> buffers;
    private LinkedHashMap<String, Field> modules;
    private SameDiff tape;

    private Tensor output;
    private Tensor[] outputs;
    static private HashMap<String, INDArray> placeHolders;
    private String name = "root";

    protected Module() {
        dataType = DataType.FLOAT;
        parameters = new LinkedHashMap<>();
        buffers = new LinkedHashMap<>();
        modules = new LinkedHashMap<>();

        for (Field f : this.getClass().getDeclaredFields()) {
            for (Annotation a : f.getDeclaredAnnotations()) {
                if (a instanceof Param) {
                    parameters.put(f.getName(), f);
                } else if (a instanceof Buff) {
                    buffers.put(f.getName(), f);
                } else if (a instanceof Mod) {
                    modules.put(f.getName(), f);
                }
            }
        }
    }

    protected Tensor forward(Tensor in) {
        throw new InvalidModuleException("Wrong or missing forward method.");
    }

    protected Tensor forward(Tensor first, Tensor second) {
        throw new InvalidModuleException("Wrong or missing forward method.");
    }

    protected Tensor[] forward(Tensor[] inputs) {
        throw new InvalidModuleException("Wrong or missing forward method.");
    }

    /**
     * Reset Parameters should be called by a module once it has finished its
     * constructor function if the module contains parameters.
     *
     * The function should call an initialization function and set the initial
     * values of it's parameters here.
     */
    protected void resetParameters() {
    }

    /**
     * Apply a function recursively to this and every submodule of this.
     *
     * Used typically for initializing parameters of a model.
     *
     * @param fn Function to be applied to each submodule.
     * @return this module instance.
     */
    public Module applyToModules(Consumer<Module> fn) {
        for (Pair<String, Module> tuple : children()) {
            String name = tuple.getKey();
            Module m = tuple.getValue();
            m.setName(getName() + "/" + name);
            m.applyToModules(fn);
        }
        fn.accept(this);

        return this;
    }

    private String getName() {
        return name;
    }

    private void setName(String name) {
        this.name = name;
    }

    public void backward() {
        if (training) {
            List<String> parameterNames = new ArrayList<>();
            for (Parameter p : parameters()) {
                if (p.isRequiresGrad()) {
                    parameterNames.add(p.name());
                }
            }
            tape.calculateGradients(placeHolders, parameterNames);
        }
    }

    public Tensor apply(final INDArray input) {
        if (dynamic || getTape() == null || output == null) {
            setTape(SameDiff.create());

            applyToModules(this::setupParameters);
            Tensor placeHolder = t(tape.placeHolder("input0", input.dataType(), input.shape()));
            output = forward(placeHolder);
        }

        return output;
    }

    /**
     * Calls this model with the input.
     * @param input Data to use as the input.
     * @return The result of the model operation.
     */
    public INDArray call(final INDArray input) {
        placeHolders = new HashMap<String, INDArray>() {{
            put("input0", input);
        }};

        output = apply(input);
        return output.eval(placeHolders);
    }


    public Tensor apply(final INDArray firstInput, final INDArray secondInput) {
        if (dynamic || getTape() == null || output == null) {
            setTape(SameDiff.create());

            applyToModules(this::setupParameters);

            Tensor placeHolder1 = t(tape.placeHolder("input0", firstInput.dataType(), firstInput.shape()));
            Tensor placeHolder2 = t(tape.placeHolder("input1", secondInput.dataType(), secondInput.shape()));
            output = forward(placeHolder1, placeHolder2);
        }

        return output;
    }

    /**
     * Calls this model with two inputs.
     * @param firstInput Data to use as the first input.
     * @param secondInput Data to use as the second input.
     * @return The result of the model operation.
     */
    public INDArray call(final INDArray firstInput, final INDArray secondInput) {
        placeHolders = new HashMap<String, INDArray>() {{
            put("input0", firstInput);
            put("input1", secondInput);
        }};

        output = apply(firstInput, secondInput);
        return output.eval(placeHolders);
    }

    public Tensor[] apply(final INDArray[] inputs) {
        if (dynamic || getTape() == null || outputs == null) {
            setTape(SameDiff.create());

            applyToModules(this::setupParameters);

            Tensor[] placeHolderVariables = new Tensor[inputs.length];
            int i = 0;
            for (INDArray input : inputs) {
                placeHolderVariables[i] = t(tape.placeHolder("input" + i, input.dataType(), input.shape()));
                i++;
            }

            outputs = forward(placeHolderVariables);
        }

        return outputs;
    }

    /**
     * Calls this model with multiple inputs and returns multiple outputs.
     * @param inputs Data to use as the inputs.
     * @return The results of the model operation.
     */
    public INDArray[] call(final INDArray[] inputs) {
        placeHolders = new HashMap<>();

        int i = 0;
        for (INDArray input : inputs) {
            placeHolders.put("input" + i, input);
            i++;
        }

        outputs = apply(inputs);


        Map<String, INDArray> outputMap = getTape().outputAll(placeHolders);
        INDArray[] results = new INDArray[outputs.length];
        i = 0;
        for (SDVariable output : outputs) {
            results[i] = outputMap.get(output.name());
            i++;
        }

        return results;
    }


    /**
     * Return a list of all modules that are a part of this module.
     * @return the ordered list of modules.
     */
    public Iterable<Pair<String, Module>> children() {
        List<Pair<String, Module>> ret = new ArrayList<>();

        for (Field f : modules.values()) {
            try {
                ret.add(Pair.of(f.getName(), (Module)f.get(this)));
            } catch (IllegalAccessException e) {
                String fieldName = f.getName();
                throw new InvalidModuleException("Field: " + fieldName + " is not a Module but annotated as one.", e);
            }
        }

        return ret;
    }

    /**
     * Sets the module training mode.
     *
     * @param shouldTrain Set whether this module should train or not.
     * @return this module instance.
     */
    public Module train(boolean shouldTrain) {
        training = shouldTrain;
        for (Pair<String, Module> tuple : children()) {
            tuple.getValue().train(shouldTrain);
        }

        return this;
    }

    /**
     * Gets the module training mode.
     *
     * @return the module's current training mode.
     */
    public boolean getTrain() {
        return training;
    }

    /**
     * Sets the module to evaluation mode.
     *
     * Only effects certain modules like Dropout or BatchNorm.
     *
     * @return this module instance.
     */
    public Module eval() {
        return train(false);
    }

    /**
     * Change if autograd is required in this module.
     *
     * Helpful for freezing part of the module for finetuning or
     * training only parts of a model individually (GANs, etc.)
     *
     * @param shouldRequire Whether autograd is necessary.
     * @return this module instance.
     */
    public Module requiresGrad(boolean shouldRequire) {
        for (Field f : parameters.values()) {
            Parameter p = null;
            try {
                p = (Parameter) f.get(this);
            } catch (IllegalAccessException e) {
                String fieldName = f.getName();
                String msg = "Field " + fieldName + " is annotated as @Param but is not a Parameter.";
                throw new InvalidModuleException(msg, e);
            }
            p.requiresGrad(shouldRequire);
        }

        return this;
    }

    /**
     * Sets the gradients of all model parameters to zero.
     */
    public void zeroGrad() {
        for (Field f : parameters.values()) {
            Parameter p = null;
            try {
                p = (Parameter) f.get(this);
            } catch (IllegalAccessException e) {
                String fieldName = f.getName();
                String msg = "Field " + fieldName + " is annotated as @Param but is not a Parameter.";
                throw new InvalidModuleException(msg, e);
            }
            if (p.hasGradient()) {
                p.gradient().setArray(Nd4j.zeros(p.gradient().getShape()));
            }
        }
    }

    /**
     * Recursively get all the parameters defined in the Module.
     */
    public Collection<Parameter> parameters() {
        List<Parameter> ret = new ArrayList<>();
        applyToModules((module -> {
            for (Field f : module.parameters.values()) {
                try {
                    f.setAccessible(true);
                    Parameter p = (Parameter)f.get(module);
                    if (p != null) {
                        ret.add(p);
                    }
                } catch (IllegalAccessException e) {
                    String fieldName = f.getName();
                    String msg = "Field " + fieldName + " is annotated as @Param but is not a Parameter.";
                    throw new InvalidModuleException(msg, e);
                }
            }
        }));

        return ret;
    }

    private void setupParameters(Module module) {
        module.setTape(tape);
        for (Map.Entry<String, Field> entry : module.parameters.entrySet()) {
            String name = entry.getKey();
            Field field = entry.getValue();

            try {
                field.setAccessible(true);
                Parameter p = (Parameter) field.get(module);
                if (p != null) {
                    Tensor variable = t(tape.var(module.getName() + "/" + name, dataType, p.getShape()));
                    variable.setArray(p.getArr());
                    p.setData(variable);
                }
            } catch (IllegalAccessException e) {
                String msg = "Field " + name + " was annotated with @Param but is not a Parameter instance.";
                throw new InvalidModuleException(msg, e);
            }
        }
    }

    protected void setTape(SameDiff sd) {
        nn = sd.nn();
        cnn = sd.cnn();
        rnn = sd.rnn();
        loss = sd.loss();
        image = sd.image();
        math = sd.math();
        bitwise = sd.bitwise();
        random = sd.random();
        tape = sd;
    }

    public SameDiff getTape() {
        return tape;
    }

    public SDVariable getOutput() {
        return output;
    }

    public Map<String, INDArray> getPlaceHolders() {
        return placeHolders;
    }

    public boolean getDynamic() {
        return dynamic;
    }

    public void setDynamic(boolean value) {
        dynamic = value;
    }
    protected Tensor t(SDVariable var) {
        return new Tensor(this, var);
    }
}
