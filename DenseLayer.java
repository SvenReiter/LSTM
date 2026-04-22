package lstm2604221130;

public class DenseLayer {
    final Func SIGMOID= x-> (float)(1 / (1 + Math.exp(-x[0])));
    final float LEARNING_RATE= 0.1f;
    private float[] input;
    private float[] output;
    private float[] biases;
    private float[][]weights;

    public void forward(float[] hT, float[][] w, float[] b) {
        this.output= matMul(w, hT);                                         //Schritt1 Input * WeightMatrix:
        //Schritt2 Bias addieren
/*
        //Ohne Aktivierungsfunktion, mit Bias
        for(int i=0; i<this.output.length; i++) {
            this.output[i]+= this.biases[i];
        }
 */
/*
        //Mit Aktivierungsfunktion, ohne Bias
        for(int i=0; i<this.output.length; i++) {
            this.output[i]= (float)SIGMOID.calc(this.output[i]);
        }
*/
        //Mit Aktivierungsfunktion, mit Bias
        for(int i=0; i<this.output.length; i++) {
            this.output[i]+= this.biases[i];
            this.output[i]= (float)SIGMOID.calc(this.output[i]);
        }

    }

    public void updateWeightsAndBiases(float[][] delLdelW, float[] delLdelB) {
        for(int i = 0; i< biases.length; i++) {
            biases[i]-=(LEARNING_RATE*delLdelB[i]);
        }
        for(int i=0; i< weights.length; i++) {
            for(int d=0; d<weights[i].length; d++) {
                weights[i][d]-=(LEARNING_RATE* delLdelW[d][i]);
            }
        }
    }

    public void updateWeightsAndBiases(float[] delLdelW, float[] delLdelB) {
        for(int i=0; i< weights.length; i++) {
            for(int d=0; d<weights[i].length; d++) {
                weights[i][d]-=(LEARNING_RATE* delLdelW[d]);
            }
        }
        for(int i = 0; i< biases.length; i++) {
            biases[i]-=(LEARNING_RATE*delLdelB[i]);
        }
    }

    /**
     * @param fehlerGradient: [als Vektor]diese Überladung ist spezialisiert auf mehrdimensionale Fehler-Gradienten
     * @param hT:= denseLayer.input
     * @param w:= */
    public float[][] backpropagate(float[] fehlerGradient, float[] hT, float[][] w) {
        /* DONE: 09.04.2026
             5. Backpropagation in das LSTM hinein
             <Welche Werte> werden                  -delLdelX bzw. matMul(transponiert w[][], fehlerGradient[])
             <von hier>                             -(als Rückgabewert) dieser Funktion
             <wodurch> nach                         -als Rückgabewert (dieser Funktion)
             <wohin> gereicht, um sie               -an den vorherigen Knoten(Im Diagram
                                                     DenseLstmArchitectureConstistsAlsoBatchNormailzationLayer.png ist
                                                     das die letzte LSTM_cell aus der vorherigen LSTM_cell-Row)
             <wie> weiterzuverarbeiten?             -Nach Prinzip des Gradientenabstieg-Verfahrens
         */

        //Google: "Rechenbeispiel Dense-Layer aus LSTM für Forwarding und Backpropagation"
        //2. Gradient der Gewichte (W)
        float[][] delLdelW= new float[fehlerGradient.length][hT.length];
        for(int fg=0; fg< fehlerGradient.length; fg++) {
            for (int i = 0; i < hT.length; i++) {
                delLdelW[fg][i] = fehlerGradient[fg] * hT[i];
            }
        }

        //3. Gradient des Bias (b)
        float[] delLdelB= fehlerGradient;
        //4. Gradient des Inputs(um den Fehler an vorherige Schichten weiterzugeben)
        float[][]wT= new float[w[0].length][w.length];
        for(int c=0; c<w[0].length; c++) {
            for(int l=0; l<w.length; l++) {
                wT[c][l]= w[l][c]*fehlerGradient[c];
            }
        }

        //5. Update weights and biases
        updateWeightsAndBiases(delLdelW, delLdelB);

        return delLdelW;    //Das Ergebnis der Backpropagation für den Input-Fluss.
    }

    /**
     * @param fehlerGradient: [als Vektor]diese Überladung ist spezialisiert auf mehrdimensionale Fehler-Gradienten
     * @param hT:= denseLayer.input
     * @param w:=
     * @return   WENN <transponateRes> diese Dimension mehr als 1 Elemente trägt,
     *           muss die aus der Multiplikation(<fehlerGradient> * <hT>) resultierende Sequenz transponiert werden.
     *           SONST Multiplikation(<fehlerGradient> * <hT>)*/
    public float[] backpropagate(float[] fehlerGradient, float[] hT, float[][] w, boolean transponateRes) {
        /* DONE: 09.04.2026
             5. Backpropagation in das LSTM hinein
             <Welche Werte> werden                  -delLdelX bzw. matMul(transponiert w[][], fehlerGradient[])
             <von hier>                             -(als Rückgabewert) dieser Funktion
             <wodurch> nach                         -als Rückgabewert (dieser Funktion)
             <wohin> gereicht, um sie               -an den vorherigen Knoten(Im Diagram
                                                     DenseLstmArchitectureConstistsAlsoBatchNormailzationLayer.png ist
                                                     das die letzte LSTM_cell aus der vorherigen LSTM_cell-Row)
             <wie> weiterzuverarbeiten?             -Nach Prinzip des Gradientenabstieg-Verfahrens
         */

        //Google: "Rechenbeispiel Dense-Layer aus LSTM für Forwarding und Backpropagation"
        //2. Gradient der Gewichte (W)
        float[][] delLdelW= new float[fehlerGradient.length][hT.length];
        for(int fg=0; fg< fehlerGradient.length; fg++) {
            for (int i = 0; i < hT.length; i++) {
                delLdelW[fg][i] = fehlerGradient[fg] * hT[i];
            }
        }

        //3. Gradient des Bias (b)
        float[] delLdelB= fehlerGradient;
        //4. Gradient des Inputs(um den Fehler an vorherige Schichten weiterzugeben)
        float[][]wT= new float[w[0].length][w.length];
        for(int c=0; c<w[0].length; c++) {
            for(int l=0; l<w.length; l++) {
                wT[c][l]= w[l][c]*fehlerGradient[c];
            }
        }

        //5. Update weights and biases
        updateWeightsAndBiases(delLdelW, delLdelB);

        if(transponateRes) {
            return delLdelW[0];    //Das Ergebnis der Backpropagation für den Input-Fluss.
        }else {
            return matMul(wT, fehlerGradient);    //Das Ergebnis der Backpropagation für den Input-Fluss.
        }
    }

    /**
     * @param y: Fehler[als Wert]
     * @param hT:= denseLayer.input
     * @param w   */
    public void backpropagate(float y, float[] hT, float[][] w) {
        /* DONE: 09.04.2026
             5. Backpropagation in das LSTM hinein
             <Welche Werte> werden                  -delLdelX bzw. matMul(transponiert w[][], fehlerGradient[])
             <von hier>                             -(als Rückgabewert) dieser Funktion
             <wodurch> nach                         -als Rückgabewert (dieser Funktion)
             <wohin> gereicht, um sie               -an den vorherigen Knoten(Im Diagram
                                                     DenseLstmArchitectureConstistsAlsoBatchNormailzationLayer.png ist
                                                     das die letzte LSTM_cell aus der vorherigen LSTM_cell-Row)
             <wie> weiterzuverarbeiten?             -Nach Prinzip des Gradientenabstieg-Verfahrens
         */
        //1. Loss nach der Aktivierung
//        float[] L= new float[]{(float)((0.5f)*Math.pow(yHut-1,2))};
        float delLdelYhut= output[0]-y;
        float aktivierungNachZ= output[0]*(1-output[0]);

        //2. Gradient der Gewichte (W)
        float fehlerSignal= delLdelYhut* aktivierungNachZ;

        float[] delLdelW= new float[hT.length];
        for(int i=0; i< hT.length; i++) {
            delLdelW[i]= fehlerSignal * hT[i];
        }

        //3. Gradient des Bias (b)
        float[] delLdelB= new float[biases.length];
        for(int i = 0; i< biases.length; i++) {
            delLdelB[i]=fehlerSignal* biases[i];
        }

        //4. Update weights and biases
        updateWeightsAndBiases(delLdelW, delLdelB);
    }

    public DenseLayer(float[]input, int outputSize) {
        this.input= input;
        this.biases = new float[outputSize];
        for(int i = 0; i< biases.length; i++) biases[i]= (float)(Math.random()*2)-1;
        weights= new float[input.length][outputSize];
        for(int l=0; l<weights.length; l++) {
            for(int c=0; c<weights[l].length; c++) {
                weights[l][c]= (float)(Math.random()*2)-1;
            }
        }
        this.output= matMul(weights, this.input);
        for(int i=0; i<this.output.length; i++) {
            this.output[i]+= this.biases[i];
            this.output[i]= (float)SIGMOID.calc(this.output[i]);
        }
    }

    /**
     * @post Output.length:= weights[].length
     * @param input
     * @param weights
     * @param biases     */
    public DenseLayer(float[]input, float[][]weights, float[] biases) {
        this.input= input;
        this.biases = biases;
        this.weights= weights;
        forward(input, weights, biases);
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
//                res[l]= res[l]+(input[c]*weights[c][l]);
                res[l]+= (input[c]*weights[c][l]);
            }
        }
        return res;
    }

    public float[][] getWeights() {
        return this.weights;
    }

    public float[] getBiases() {
        return this.biases;
    }
}
