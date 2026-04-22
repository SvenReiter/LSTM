package lstm2604221130;

import java.util.Arrays;

public class Gate {
    private Func func;
    private float[] input;
    private float[] result;
    private float[][] weights;
    private float[] bias;
    private float[] z;

    /**
     *
     * @param func
     * @param input concatenated version of hs[] and inputX[] at time [t]
     * @param weights the gate's weights at time [t]
     * @param bias the gate's bias at time [t] */
    public Gate(Func func, float[] input, float[][] weights, float[] bias, float u, float[]preHS, boolean mulMitGetrimmtemInput) {
        this.func = func;
        this.input = input;
        this.weights = weights;
        this.bias = bias;
        calculateResult(u, preHS, mulMitGetrimmtemInput);
    }

    /**
     * @param u
     * @param preHS
     * @param mulMitGetrimmtemInput for testing aspects */
    public void calculateResult(float u, float[]preHS, boolean mulMitGetrimmtemInput) {
        if(mulMitGetrimmtemInput) {
            result = matMul(weights, Arrays.copyOfRange(input, preHS.length, input.length), mulMitGetrimmtemInput, preHS.length);
        }else {
            result= matMul(weights, input, mulMitGetrimmtemInput, preHS.length);
            z= matMul(weights, input, mulMitGetrimmtemInput, preHS.length);
        }
        float sum=0f;
        for(int i=0; i<result.length; i++) {
            /* TODO: 16.04.2026  Rekurrenten Anteil berechnen (<U*> * vorherigen HiddenState(h[t-1])):
                    Das bedingt, dass <Gate()> entspr. <U*> und diesen <HiddenState(h[t-1])> kennt
                    Dann lässt sich hier implementieren:
                        result[i]+= <U*> * <HiddenState(h[t-1])>
             */
            /*Das muss man bei deisem Beispiel nicht machen,
             *  weil wir hier mit dem konkatenierten Vektor rechnen(dieser enthält bereits den rekurrenten Anteil).
             *  Die Gewichtsvektoren implizieren alle einen 3. Wert, dieser Wert steht stellvertretend für <U>
             */
//            for(int j=0; j< preHS.length; j++) {
//                sum+= preHS[j]; //muss Skalarprodukt über den gesamten Vektor preHS[] sein
//            }
            result[i]+= u* sum;
            result[i]+= bias[i];        // bias gets added to result before proceeding activation-function
            result[i]= (float)func.calc(result[i]);
        }
    }

    public void setBias(float[]delBias) {
        for(int i=0; i<this.bias.length; i++) {
            bias[i]-= delBias[i];   //alter Bias - BiasGradient
        }

    }

    /**
     * @param weights
     * @param input
     * @pre weights[].length == input.length
     * @return  weights[][] und input[] nach Falk-Schema miteinander multiplizieren */
    private float[] matMul(float[][]weights, float[]input, boolean mulMitGetrimmtemInput, int preHSLength) {
        float[]res= new float[weights.length];
        for(int l=0; l<weights.length; l++) {
            for(int c=0; c<((mulMitGetrimmtemInput)?weights[l].length-preHSLength:weights[l].length); c++) {
//            for(int c=0; c<weights[l].length; c++) {
                res[l]+= weights[l][c] * input[c];
            }
        }
        return res;
    }

    public float[] getResult() {
        return result;
    }
    public float[] getZ() {
        return z;
    }
}
