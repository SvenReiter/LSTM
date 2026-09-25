package lstm2609250959_BPTT_postActUpreActDuerfenNichtVermischtSein;

@FunctionalInterface
public interface LossFunc {
    float[] calc(float[]hs, float[]y);
}
