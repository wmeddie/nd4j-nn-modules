package org.nd4j.autodiff.eager;

import lombok.NonNull;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.nn.Module;
import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;

import java.util.Arrays;
import java.util.Map;

/**
 * Tensor is a eager version of a SDVariable.
 */
public class Tensor extends SDVariable {
    private final SDVariable data;
    private final Module parent;
    private long[] shape;

    public Tensor(Module parent, SDVariable data) {
        this.parent = parent;
        this.data = data;
    }


    @Override
    public String name() {
        return data.name();
    }

    @Override
    public boolean isPlaceHolder() {
        return data.isPlaceHolder();
    }

    @Override
    public boolean isConstant() {
        return data.isConstant();
    }

    @Override
    public INDArray getArr() {
        return data.getArr();
    }

    @Override
    public INDArray getArr(boolean enforceExistence) {
        return data.getArr(enforceExistence);
    }

    @Override
    public SDVariable gradient() {
        return data.gradient();
    }

    @Override
    public SDVariable getGradient() {
        return data.getGradient();
    }

    @Override
    public long[] getShape() {
        try {
            shape = data.getShape();
            if (shape == null) {
                SDVariable var = data.shape();
                shape = var.eval(parent.getPlaceHolders()).toLongVector();
            }
            return shape;
        } catch (Exception ignored) {
            return null;
        }
    }

    @Override
    public void setShape(long... shape) {
        shape = shape;
        data.setShape(shape);
    }

    @Override
    public long[] placeholderShape() {
        return data.placeholderShape();
    }

    @Override
    public DataType dataType() {
        return data.dataType();
    }

    @Override
    public LongShapeDescriptor getShapeDescriptor() {
        return data.getShapeDescriptor();
    }

    @Override
    public SDVariable castTo(@NonNull DataType dataType) {
        return data.castTo(dataType);
    }

    @Override
    public SDVariable castTo(String name, @NonNull DataType dataType) {
        return data.castTo(name, dataType);
    }

    @Override
    public SDVariable dup() {
        return data.dup();
    }

    @Override
    public SDVariable assign(Number value) {
        return data.assign(value);
    }

    @Override
    public SDVariable neg() {
        return data.neg();
    }

    @Override
    public SDVariable neg(String name) {
        return data.neg(name);
    }

    @Override
    public SDVariable lt(double value) {
        return data.lt(value);
    }

    @Override
    public SDVariable lt(String name, double value) {
        return data.lt(name, value);
    }

    @Override
    public SDVariable lte(double value) {
        return data.lte(value);
    }

    @Override
    public SDVariable lte(String name, double value) {
        return data.lte(name, value);
    }

    @Override
    public SDVariable gt(double value) {
        return data.gt(value);
    }

    @Override
    public SDVariable gt(String name, double value) {
        return data.gt(name, value);
    }

    @Override
    public SDVariable gte(double value) {
        return data.gte(value);
    }

    @Override
    public SDVariable gte(String name, double value) {
        return data.gte(name, value);
    }

    @Override
    public SDVariable eq(double value) {
        return data.eq(value);
    }

    @Override
    public SDVariable eq(String name, double value) {
        return data.eq(name, value);
    }

    @Override
    public SDVariable neq(double value) {
        return data.neq(value);
    }

    @Override
    public SDVariable neq(String name, double value) {
        return data.neq(name, value);
    }

    @Override
    public SDVariable lt(SDVariable other) {
        return data.lt(other);
    }

    @Override
    public SDVariable lt(String name, SDVariable other) {
        return data.lt(name, other);
    }

    @Override
    public SDVariable lte(SDVariable other) {
        return data.lte(other);
    }

    @Override
    public SDVariable lte(String name, SDVariable other) {
        return data.lte(name, other);
    }

    @Override
    public SDVariable gt(SDVariable other) {
        return data.gt(other);
    }

    @Override
    public SDVariable gt(String name, SDVariable other) {
        return data.gt(name, other);
    }

    @Override
    public SDVariable gte(SDVariable other) {
        return data.gte(other);
    }

    @Override
    public SDVariable gte(String name, SDVariable other) {
        return data.gte(name, other);
    }

    @Override
    public SDVariable eq(SDVariable other) {
        return data.eq(other);
    }

    @Override
    public SDVariable eq(String name, SDVariable other) {
        return data.eq(name, other);
    }

    @Override
    public SDVariable neq(SDVariable other) {
        return data.neq(other);
    }

    @Override
    public SDVariable neq(String name, SDVariable other) {
        return data.neq(name, other);
    }

    @Override
    public SDVariable mmul(SDVariable other) {
        return data.mmul(other);
    }

    @Override
    public SDVariable mmul(String name, SDVariable other) {
        return data.mmul(name, other);
    }

    @Override
    public SDVariable mmul(String name, SDVariable other, @NonNull MMulTranspose mMulTranspose) {
        return data.mmul(name, other, mMulTranspose);
    }

    @Override
    public SDVariable dot(SDVariable other, int... dimensions) {
        return data.dot(other, dimensions);
    }

    @Override
    public SDVariable dot(String name, SDVariable other, int... dimensions) {
        return data.dot(name, other, dimensions);
    }

    @Override
    public SDVariable add(double scalar) {
        return data.add(scalar);
    }

    @Override
    public SDVariable add(String varName, double scalar) {
        return data.add(varName, scalar);
    }

    @Override
    public SDVariable add(SDVariable other) {
        return data.add(other);
    }

    @Override
    public SDVariable add(String name, SDVariable x) {
        return data.add(name, x);
    }

    @Override
    public SDVariable plus(SDVariable other) {
        return data.plus(other);
    }

    @Override
    public SDVariable plus(double other) {
        return data.plus(other);
    }

    @Override
    public SDVariable sub(double scalar) {
        return data.sub(scalar);
    }

    @Override
    public SDVariable sub(String varName, double scalar) {
        return data.sub(varName, scalar);
    }

    @Override
    public SDVariable sub(SDVariable x) {
        return data.sub(x);
    }

    @Override
    public SDVariable sub(String name, SDVariable x) {
        return data.sub(name, x);
    }

    @Override
    public SDVariable minus(SDVariable other) {
        return data.minus(other);
    }

    @Override
    public SDVariable minus(double other) {
        return data.minus(other);
    }

    @Override
    public SDVariable div(double scalar) {
        return data.div(scalar);
    }

    @Override
    public SDVariable div(String varName, double scalar) {
        return data.div(varName, scalar);
    }

    @Override
    public SDVariable div(SDVariable x) {
        return data.div(x);
    }

    @Override
    public SDVariable div(String name, SDVariable x) {
        return data.div(name, x);
    }

    @Override
    public SDVariable fdiv(String name, SDVariable x) {
        return data.fdiv(name, x);
    }

    @Override
    public SDVariable mod(String name, SDVariable x) {
        return data.mod(name, x);
    }

    @Override
    public SDVariable mul(double scalar) {
        return data.mul(scalar);
    }

    @Override
    public SDVariable mul(String varName, double scalar) {
        return data.mul(varName, scalar);
    }

    @Override
    public SDVariable mul(SDVariable x) {
        return data.mul(x);
    }

    @Override
    public SDVariable mul(String name, SDVariable x) {
        return data.mul(name, x);
    }

    @Override
    public SDVariable times(SDVariable other) {
        return data.times(other);
    }

    @Override
    public SDVariable times(double other) {
        return data.times(other);
    }

    @Override
    public SDVariable pow(double scalar) {
        return data.pow(scalar);
    }

    @Override
    public SDVariable pow(String varName, double scalar) {
        return data.pow(varName, scalar);
    }

    @Override
    public SDVariable rsub(double scalar) {
        return data.rsub(scalar);
    }

    @Override
    public SDVariable rsub(String varName, double scalar) {
        return data.rsub(varName, scalar);
    }

    @Override
    public SDVariable rsub(SDVariable x) {
        return data.rsub(x);
    }

    @Override
    public SDVariable rsub(String name, SDVariable x) {
        return data.rsub(name, x);
    }

    @Override
    public SDVariable rdiv(double scalar) {
        return data.rdiv(scalar);
    }

    @Override
    public SDVariable rdiv(String varName, double scalar) {
        return data.rdiv(varName, scalar);
    }

    @Override
    public SDVariable rdiv(SDVariable sameDiffVariable) {
        return data.rdiv(sameDiffVariable);
    }

    @Override
    public SDVariable rdiv(String name, SDVariable x) {
        return data.rdiv(name, x);
    }

    @Override
    public SDVariable squaredDifference(SDVariable x) {
        return data.squaredDifference(x);
    }

    @Override
    public SDVariable squaredDifference(String name, SDVariable x) {
        return data.squaredDifference(name, x);
    }

    @Override
    public SDVariable sum(int... dimensions) {
        return data.sum(dimensions);
    }

    @Override
    public SDVariable sum(boolean keepDims, int... dimensions) {
        return data.sum(keepDims, dimensions);
    }

    @Override
    public SDVariable sum(String name, int... dimensions) {
        return data.sum(name, dimensions);
    }

    @Override
    public SDVariable sum(String name, boolean keepDims, int... dimensions) {
        return data.sum(name, keepDims, dimensions);
    }

    @Override
    public SDVariable mean(boolean keepDims, int... dimensions) {
        return data.mean(keepDims, dimensions);
    }

    @Override
    public SDVariable mean(String name, int... dimensions) {
        return data.mean(name, dimensions);
    }

    @Override
    public SDVariable mean(int... dimensions) {
        return data.mean(dimensions);
    }

    @Override
    public SDVariable mean(String name, boolean keepDims, int... dimensions) {
        return data.mean(name, keepDims, dimensions);
    }

    @Override
    public SDVariable std(boolean biasCorrected, int... dimensions) {
        return data.std(biasCorrected, dimensions);
    }

    @Override
    public SDVariable std(String name, boolean biasCorrected, int... dimensions) {
        return data.std(name, biasCorrected, dimensions);
    }

    @Override
    public SDVariable std(String name, boolean biasCorrected, boolean keepDims, int... dimensions) {
        return data.std(name, biasCorrected, keepDims, dimensions);
    }

    @Override
    public SDVariable prod(int... dimensions) {
        return data.prod(dimensions);
    }

    @Override
    public SDVariable prod(boolean keepDims, int... dimensions) {
        return data.prod(keepDims, dimensions);
    }

    @Override
    public SDVariable prod(String name, int... dimensions) {
        return data.prod(name, dimensions);
    }

    @Override
    public SDVariable prod(String name, boolean keepDims, int... dimensions) {
        return data.prod(name, keepDims, dimensions);
    }

    @Override
    public SDVariable min(int... dimensions) {
        return data.min(dimensions);
    }

    @Override
    public SDVariable min(boolean keepDims, int... dimensions) {
        return data.min(keepDims, dimensions);
    }

    @Override
    public SDVariable min(String name, int... dimensions) {
        return data.min(name, dimensions);
    }

    @Override
    public SDVariable min(String name, boolean keepDims, int... dimensions) {
        return data.min(name, keepDims, dimensions);
    }

    @Override
    public SDVariable max(int... dimensions) {
        return data.max(dimensions);
    }

    @Override
    public SDVariable max(String name, int... dimensions) {
        return data.max(name, dimensions);
    }

    @Override
    public SDVariable max(boolean keepDims, int... dimensions) {
        return data.max(keepDims, dimensions);
    }

    @Override
    public SDVariable max(String name, boolean keepDims, int... dimensions) {
        return data.max(name, keepDims, dimensions);
    }

    @Override
    public SDVariable norm1(int... dimensions) {
        return data.norm1(dimensions);
    }

    @Override
    public SDVariable norm1(boolean keepDims, int... dimensions) {
        return data.norm1(keepDims, dimensions);
    }

    @Override
    public SDVariable norm1(String name, int... dimensions) {
        return data.norm1(name, dimensions);
    }

    @Override
    public SDVariable norm1(String name, boolean keepDims, int... dimensions) {
        return data.norm1(name, keepDims, dimensions);
    }

    @Override
    public SDVariable norm2(int... dimensions) {
        return data.norm2(dimensions);
    }

    @Override
    public SDVariable norm2(boolean keepDims, int... dimensions) {
        return data.norm2(keepDims, dimensions);
    }

    @Override
    public SDVariable norm2(String name, int... dimensions) {
        return data.norm2(name, dimensions);
    }

    @Override
    public SDVariable norm2(String name, boolean keepDims, int... dimensions) {
        return data.norm2(name, keepDims, dimensions);
    }

    @Override
    public SDVariable normmax(int... dimensions) {
        return data.normmax(dimensions);
    }

    @Override
    public SDVariable normmax(boolean keepDims, int... dimensions) {
        return data.normmax(keepDims, dimensions);
    }

    @Override
    public SDVariable normmax(String name, int... dimensions) {
        return data.normmax(name, dimensions);
    }

    @Override
    public SDVariable normmax(String name, boolean keepDims, int... dimensions) {
        return data.normmax(name, keepDims, dimensions);
    }

    @Override
    public SDVariable argmax(int... dimensions) {
        return data.argmax(dimensions);
    }

    @Override
    public SDVariable argmax(String name, int... dimensions) {
        return data.argmax(name, dimensions);
    }

    @Override
    public SDVariable argmax(String name, boolean keepDims, int... dimensions) {
        return data.argmax(name, keepDims, dimensions);
    }

    @Override
    public SDVariable argmin(int... dimensions) {
        return data.argmin(dimensions);
    }

    @Override
    public SDVariable argmin(String name, int... dimensions) {
        return data.argmin(name, dimensions);
    }

    @Override
    public SDVariable argmin(String name, boolean keepDims, int... dimensions) {
        return data.argmin(name, keepDims, dimensions);
    }

    @Override
    public SDVariable shape() {
        return data.shape();
    }

    @Override
    public SDVariable rank() {
        return data.rank();
    }

    @Override
    public SDVariable reshape(SDVariable newShape) {
        return data.reshape(newShape);
    }

    @Override
    public SDVariable reshape(int... newShape) {
        return data.reshape(newShape);
    }

    @Override
    public SDVariable reshape(long... newShape) {
        return data.reshape(newShape);
    }

    @Override
    public SDVariable permute(int... dimensions) {
        return data.permute(dimensions);
    }

    @Override
    public SDVariable permute(SDVariable dimensions) {
        return data.permute(dimensions);
    }

    @Override
    public SDVariable setArray(INDArray array) {
        return data.setArray(array);
    }

    @Override
    public INDArray eval() {
        return data.eval();
    }

    @Override
    public INDArray eval(Map<String, INDArray> placeholders) {
        return data.eval(placeholders);
    }

    @Override
    public String toString() {
        String arr;
        String shape;
        try {
            shape = Arrays.toString(getShape());
        } catch (Exception ignored) {
            shape = "[???]";
        }

        try {
            arr = data.getArr().toString();
        } catch (Exception ignored) {
            try {
                data.eval(parent.getPlaceHolders());
                arr = data.getArr().toString();
            } catch (Exception alsoIgnored) {
                arr  = "[???]";
            }
        }

        return "tensor(shape=" + shape + ", array=" + arr + ")";
    }

    @Override
    public void addControlDependency(SDVariable controlDependency) {
        data.addControlDependency(controlDependency);
    }

    @Override
    public SDVariable get(SDIndex... indices) {
        return data.get(indices);
    }

    @Override
    public SDVariable convertToConstant() {
        return data.convertToConstant();
    }

    @Override
    public SDVariable convertToVariable() {
        return data.convertToVariable();
    }

    @Override
    public SDVariable rename(String newName) {
        return data.rename(newName);
    }

    @Override
    public void markAsLoss() {
        data.markAsLoss();
    }

    @Override
    public boolean hasGradient() {
        return data.hasGradient();
    }

    @Override
    public boolean equals(Object o) {
        return data.equals(o);
    }

    @Override
    public int hashCode() {
        return data.hashCode();
    }

    @Override
    public SDVariable clone(SameDiff sd) {
        return data.clone(sd);
    }

    @Override
    public SameDiff getSameDiff() {
        return data.getSameDiff();
    }

    @Override
    public DifferentialFunction getCreator() {
        return data.getCreator();
    }

    @Override
    public void setSameDiff(SameDiff sameDiff) {
        data.setSameDiff(sameDiff);
    }

    @Override
    public void setCreator(DifferentialFunction creator) {
        data.setCreator(creator);
    }

    @Override
    public void setVarName(String varName) {
        data.setVarName(varName);
    }

    @Override
    public VariableType getVariableType() {
        return data.getVariableType();
    }

    @Override
    public void setVariableType(VariableType variableType) {
        data.setVariableType(variableType);
    }

    @Override
    public void setDataType(DataType dataType) {
        data.setDataType(dataType);
    }
}
