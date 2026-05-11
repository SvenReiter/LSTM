package lstm2605111326;

import java.util.ArrayList;

public class Gate {
    private Func func;
    private float[] input;
    private ArrayList<float[]> result;
    private float[][] weights;
    private float[] bias;
    private float[] z;

    /**
     *
     * @param func
     * @param input concatenated version of hs[] and inputX[] at time [t]
     * @param weights the gate's weights at time [t]
     * @param bias the gate's bias at time [t]
     * @param t timestep */
    public Gate(Func func, float[] input, float[][] weights, float[] bias, float u, float[]preHS, int t) {
        this.func = func;
        this.input = input;
        this.weights = weights;
        this.bias = bias;
        this.result= new ArrayList<>();
        calculateResult(u, preHS, t);
    }

    /**
     * @param u
     * @param preHS
     * @param t timestep */
    public void calculateResult(float u, float[]preHS, int t) {
        result.add(new float[]{weights[0][0]* input[t]});
        this.z= new float[result.get(t).length];
        for(int i=0; i<result.get(t).length; i++) {
            result.get(t)[i]+= u* preHS[i];
            result.get(t)[i]+= bias[i];        // bias gets added to result before proceeding activation-function
            z[i]= result.get(t)[i];
            result.get(t)[i]= (float)func.calc(result.get(t)[i]);
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
     * @param t timestep
     * @pre weights[].length == input.length
     * @return  weights[][] und input[] nach Falk-Schema miteinander multiplizieren */
    private float[] matMul(float[][]weights, float[]input, boolean mulMitGetrimmtemInput, int preHSLength, int t) {
        float[]res= new float[weights.length];
        res[0]= weights[0][0]* input[t];
        /*
        for(int l=0; l<weights.length; l++) {
//            for(int c=0; c<((mulMitGetrimmtemInput)?weights[l].length-preHSLength:weights[l].length); c++) {
            for(int c=0; c<weights[l].length; c++) {
                res[l]+= weights[l][c] * input[c];
            }
        }
         */
        return res;
    }

    /**
     * @param t timestep
     * @return the result at timestep t */
    public float[] getResult(int t) {
        return result.get(t);
    }
    public float[] getZ() {
        return z;
    }
}
