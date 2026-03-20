package lstm2603200945;

public class DenseLayer {
    final Func SIGMOID= x-> (float)(1 / (1 + Math.exp(-x[0])));
    final Func SIGMOID_DERIVATIVE= x->(SIGMOID.calc(x)*(1-SIGMOID.calc(x)));
    /*todo
        Ein DenseLayer bildet ein FullyConnectedLayer ab
        Input:          Output of Hidden-/BatchNormalization-Layer  [size N]
        Output:         An arbitrary size [size M]
        Weight matrix:  connects given N Inputs with M Outputs [size NxM]
        Bias:           M Bias-values
        *
        Bsp:
            Input:          [1.0, 0.5, -0.2]
            Output:         M:=2
                                \__
                               /   \
            WeightMatrix:   {{0.2, 0.8},
                             {0.5, -0.1},
                             {0.1, 0.3}}
            Bias:           {0.1, 0.2}
        *
            Schritt1 Input * WeightMatrix:
                Neuron1: (1.0*0.2)+(0.5*0.5)+(-0.2*0.1) =0.2+0.25-0.02=0.43= 0.43
                Neuron2: (1.0*0.8)+(0.5*-0.1)+(-0.2*0.3)=0.8-0.05-0.06=0.69= 0.69
            Schritt2 Bias addieren
                Output1: 0.43+0.1= 0.53
                Output2: 0.69+0.2= 0.89
        *
            Ergebnis: Der Dense-Layer gibt den Vektor(0.53, 0.89) aus.
                      Auf diesen wird oft noch eine Aktivierungsfunktion wie Sigmoid/ReLU/Softmax angewendet.
     */
    private float[] input;
    private float[] output;
    private float[] bias;
    private float[][]weights= new float[][]{
            new float[]{0.2f, 0.5f, 0.1f},
            new float[]{0.8f, -0.1f, 0.3f}};

    public DenseLayer(float[]input, int outputSize) {
        this.input= input;
        this.bias= new float[outputSize];
        for(int i=0; i< bias.length; i++)bias[i]= (float)(Math.random()*2)-1;
        weights= new float[input.length][outputSize];
        for(int l=0; l<weights.length; l++) {
            for(int c=0; c<weights[l].length; c++) {
                weights[l][c]= (float)(Math.random()*2)-1;
            }
        }
        this.output= matMul(weights, this.input);
        for(int i=0; i<this.output.length; i++) {
            this.output[i]+= this.bias[i];
            this.output[i]= (float)SIGMOID.calc(this.output[i]);
        }
    }

    /**
     * @post Output.length:= weights[].length
     * @param input
     * @param weights
     * @param bias     */
    public DenseLayer(float[]input, float[][]weights, float[]bias) {                                                    //For testing
        this.input= input;
        this.bias= bias;
        //Schritt1 Input * WeightMatrix:
        this.output= matMul(weights, this.input);
        //Schritt2 Bias addieren
        for(int i=0; i<this.output.length; i++) {
            this.output[i]+= this.bias[i];
        }/*
        //Auf diesen wird oft noch eine Aktivierungsfunktion wie Sigmoid/ReLU/Softmax angewendet.
        for(int i=0; i<this.output.length; i++) {
            this.output[i]= (float)SIGMOID.calc(this.output[i]);
        }*/
    }

    public float[] getOutput() {
        return output;
    }

    /**
     * @param weights
     * @param input
     * @pre weights[].length == input.length
     * @return  weights[][] und input[] nach Falk-Schema miteinander multiplizieren */
    private float[] matMul(float[][]weights, float[]input) {
        float[]res= new float[weights[0].length];
        for(int l=0; l<weights[0].length; l++) {
            for(int c=0; c<input.length; c++) {
                res[l]+= input[c]*weights[c][l];
            }
        }
        return res;
    }
}
