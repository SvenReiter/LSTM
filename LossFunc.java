package lstm2609251410;

@FunctionalInterface
public interface LossFunc {
    float[] calc(float[]hs, float[]y);
}
