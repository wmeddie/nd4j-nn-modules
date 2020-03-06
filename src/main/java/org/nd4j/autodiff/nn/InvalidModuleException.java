package org.nd4j.autodiff.nn;

public class InvalidModuleException extends RuntimeException {
    public InvalidModuleException(String message, Throwable t) {
        super(message, t);
    }

    public InvalidModuleException(String message) {
        super(message);
    }
}
