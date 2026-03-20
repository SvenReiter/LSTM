package lstm2603200945;

public class Gate {
    private Func func;
    private float[] input;
    private float[] result;
    private float[][] weights;
    private float[] bias;

    /**
     *
     * @param func
     * @param input concatenated version of hs[] and inputX[] at time [t]
     * @param weights the gate's weights at time [t]
     * @param bias the gate's bias at time [t] */
    public Gate(Func func, float[] input, float[][] weights, float[] bias) {
        this.func = func;
        this.input = input;
        this.weights = weights;
        this.bias = bias;
        calculateResult();
    }

    public void calculateResult() {
        result= matMul(weights, input);
        for(int i=0; i<result.length; i++) {
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
    private float[] matMul(float[][]weights, float[]input) {
        float[]res= new float[weights.length];
        for(int l=0; l<weights.length; l++) {
            for(int c=0; c<weights[l].length; c++) {
                res[l]+= weights[l][c] * input[c];
            }
        }
        return res;
    }

    public float[] getResult() {
        return result;
    }
}
