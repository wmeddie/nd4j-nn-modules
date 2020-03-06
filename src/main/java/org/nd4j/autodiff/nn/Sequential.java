package org.nd4j.autodiff.nn;

import org.nd4j.autodiff.eager.Tensor;
import org.nd4j.autodiff.samediff.SDVariable;

import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedHashMap;

/**
 * A sequential model.
 *
 * Modules will be added to it in the order they are passed in the constructor.
 */
public class Sequential extends Module implements Collection<Module> {
    private LinkedHashMap<String, Module> submodules;

    /**
     * Create a sequential model of the modules.
     *
     * @param modules the modules to add.
     */
    public Sequential(Module... modules) {
        super();
        submodules = new LinkedHashMap<>();
        int layer = 0;
        for (Module m : modules) {
            submodules.put("Layer" + layer, m);
        }
        setDynamic(false);
    }

    /**
     * Create a sequential model of the modules with custom names.
     *
     * @param modules A map of modules to add.
     */
    public Sequential(LinkedHashMap<String, Module> modules) {
        submodules = modules;
    }

    @Override
    protected Tensor forward(Tensor in) {
        Tensor x = in;
        for (Module m : submodules.values()) {
            x = m.forward(x);
        }

        return x;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("[\n");

        for (String name : submodules.keySet()) {
            sb.append(name);
            sb.append(":");
            sb.append(submodules.get(name));
            sb.append("\n");
        }

        return sb.toString();
    }

    // Region: Collection Methods
    @Override
    public int size() {
        return submodules.size();
    }

    @Override
    public boolean isEmpty() {
        return submodules.isEmpty();
    }

    @Override
    public boolean contains(Object o) {
        return submodules.containsValue(o);
    }

    @Override
    public Iterator<Module> iterator() {
        return submodules.values().iterator();
    }

    @Override
    public Object[] toArray() {
        return submodules.values().toArray();
    }

    @Override
    public <T> T[] toArray(T[] ts) {
        return submodules.values().toArray(ts);
    }

    @Override
    public boolean add(Module module) {
        submodules.put("layer" + submodules.size(), module);
        return true;
    }

    @Override
    public boolean remove(Object o) {
        return false;
    }

    @Override
    public boolean containsAll(Collection<?> collection) {
        return submodules.values().containsAll(collection);
    }

    @Override
    public boolean addAll(Collection<? extends Module> collection) {
        for (Module m : collection) {
            add(m);
        }
        return true;
    }

    @Override
    public boolean removeAll(Collection<?> collection) {
        return false;
    }

    @Override
    public boolean retainAll(Collection<?> collection) {
        return false;
    }

    @Override
    public void clear() {
        submodules.clear();
    }
}
