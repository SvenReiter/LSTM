package lstm2604131306;

public class LSTM_cell {
    LSTM_cell prevCell, nextCell;
    final Func SIGMOID= x-> (float)(1 / (1 + Math.exp(-x[0])));
    final Func SIGMOID_DERIVATIVE= x->(SIGMOID.calc(x)*(1-SIGMOID.calc(x)));
    final Func TANH= x-> (float)((Math.exp(x[0])-Math.exp(-x[0])) / (Math.exp(x[0])+Math.exp(-x[0])));
    final Func TANH_DERIVATIVE= x->(1.0 - (TANH.calc(x)*TANH.calc(x)));
    final Gate forgetGate;                                                                                              //Entscheidet, was vergessen wird.
    final Gate inputGate;                                                                                               //Entscheidet, welche neuen Informationen gespeichert werden.
    final Gate candidateGate;                                                                                           //Erstellt neue Kandidatenwerte.
    final Gate outputGate;                                                                                              //Entscheidet, was ausgegeben wird
    final float ALPHA= 0.000001f;                                                                                       //Lernrate
    float[][] wF,wI,wC,wO;
    float[][] delLdelWf, delLdelWi, delLdelWc, delLdelWo, delLdelWy;
    float[] bF, bI, bC, bO;                                                                                             //Eine Initialisierung auf 1 sorgt dafür, dass Informationen ungehindert durch die Zeit fließen, was das Lernen von Langzeitabhängigkeiten massiv erleichtert.
    float[] delLbF, delLbI, delLbC, delLbO;
    float[] hiddenState, cellState;
    float[] z,                                                                                                          /*"The raw output of the neural network, also known as Logit."*/
            yHat,                                                                                                       /*Resulting output*/
            y,                                                                                                          /*Assuming the output value for the input(xt:=){1,0,0} is
                                                                                                                         * {0,1,0}(the error needed for BPTT is(yHat-y))             */
            delLdelZ,                                                                                                   //Resulting loss
            xt,
            Xt;
    /*the last z[t] layer is used like a kind of fully-connected layer (10:17)
     *  .
     *  Also erscheinen hier mehrere dieser <LSTM_node>s [fully-connected] und und folgend ein fully-connected layer.
     *  Anschließend folgt ein SOFTMAX-Layer um die Wahrscheinlichkeiten für die Fehlerberechnung ausgehend von
     *  den Werten aus dem fully-connected layer zu bestimmen.
     */

    /**
     * @param preCs the currents nodes previous[t-1] CS-value
     * @param preHs the currents nodes previous[t-1] HS-value
     * @param xt current[t] input
     * @post concatenated version of preHs[] and inputX[]
     *       The concatenated version depicts the assumed input.
     * @param y expected result
     * @param wF,wI,wC,wO internal weights-matrices
     * @param bF,bI,bC,bO bias-matrices */
    public LSTM_cell(float[]preCs, float[]preHs, float[]xt, float[]y, float[][]wF, float[][]wI, float[][]wC, float[][]wO,
                     float[]bF, float[]bI, float[]bC, float[]bO) {
        hiddenState = preHs;
        cellState = preCs;
        this.wF= wF;
        this.wI= wI;
        this.wC= wC;
        this.wO= wO;
        this.xt= xt;
        this.y= y;
        this.bF= bF;
        this.bI= bI;
        this.bC= bC;
        this.bO= bO;
        delLbF= new float[bF.length];
        delLbI= new float[bI.length];
        delLbC= new float[bC.length];
        delLbO= new float[bO.length];
        delLdelZ= new float[y.length];
        Xt= new float[preHs.length+ xt.length];
        for(int i=0; i<preHs.length; i++) {
            Xt[i]=preHs[i];
        }
        for(int i=0; i<xt.length; i++) {
            Xt[preHs.length+i]=xt[i];
        }
        forgetGate= new Gate(SIGMOID, Xt, wF, bF);
        inputGate= new Gate(SIGMOID, Xt, wI, bI);
        candidateGate= new Gate(TANH, Xt, wC, bC);
        outputGate= new Gate(SIGMOID, Xt, wO, bO);
    }

    public float[] dCSt() {
        return cellState;
    }

    public float[] dHSt() {
        return hiddenState;
    }

    public float[][] getDelLdelW(char of) throws IllegalArgumentException{
        switch(of) {
            case 'f':
                return delLdelWf;
            case 'i':
                return delLdelWi;
            case 'c':
                return delLdelWc;
            case 'o':
                return delLdelWo;
            case 'y':
                return delLdelWy;
        }
        throw new IllegalArgumentException();
    }

    /**
     * @param z the neuronalNets logit/final output
     * @return softmax-result of     */
    float[]softmaxBeta(float[]z) {
        float[]result= new float[z.length];
        for(int i=0; i<z.length; i++) {
            result[i]= (float)Math.pow(Math.E, z[i]);
        }
        float sumOf= 0;
        for(int i=0; i< result.length; i++) {
            sumOf+= result[i];
        }
        for(int i=0; i< result.length; i++) {
            result[i]/= sumOf;
        }
        return result;
    }


    /**
     * another softmax-implementation
     * @param logits
     * @return softmax-result of     */
    float[] softmax(float[] logits) {
        float max=0;
        for(int i=0; i< logits.length; i++) {
            if(max< logits[i]) {
                max= logits[i];
            }
        }
        float[]exp_values= new float[logits.length];    //Numerator
        float sumExpValues= 0;                          //Denominator
        for(int i=0; i< exp_values.length; i++) {
            exp_values[i]= (float)Math.exp(logits[i]- max);
            sumExpValues+= exp_values[i];
        }
        float[]probabilities= new float[logits.length];
        for(int i=0; i< probabilities.length; i++) {
            probabilities[i]= exp_values[i]/sumExpValues;
        }
        return probabilities;
    }

    /**
     * @param yHat, y
     * @return dl/dZT */
    public void calcLoss(float[]yHat, float[]y) {
        float[] result= new float[delLdelZ.length];
        for(int i=0; i<3; i++) {
            result[i]= yHat[i]-y[i];                                                                                    //soft Max and cross entropy are used YT hat minus y becomes dl/dZT
        }
        delLdelZ= result;
    }

    /**
     * @param wY size of output/logit    */
    public void forward(float[][]wY) {
        // sowohl cellState[] als auch hiddenState[] müssen vom Resultat der vorherigen Zelle übernommen werden
        hiddenState= prevCell.dHSt();
        cellState= prevCell.dCSt();
        for(int i=0; i<cellState.length; i++) {
            cellState[i]*=forgetGate.getResult()[i];
        }
        for(int i=0; i<cellState.length; i++) {
            cellState[i]+=inputGate.getResult()[i] * candidateGate.getResult()[i];
        }
        for(int i=0; i<hiddenState.length; i++) {
            hiddenState[i]= (float) (TANH.calc(cellState[i]) * outputGate.getResult()[i]);
        }
        z= new float[wY.length];            //<z> ist public; often called raw values; also known as Logit.
        for(int l=0; l<wY.length; l++) {
            for(int c=0; c<hiddenState.length; c++) {
                z[l]+= wY[l][c] * hiddenState[c];
            }
        }
        yHat= softmax(z);
    }

    /**
     * @param xt    the input[t]
     * @param Xt    the concatenation of xt and HS[t-1]
     * @param wY    */
    public void backpropagate(float[]xt, float[]Xt, float[][]wY) {
        float[]lMulWy= new float[outputGate.getResult().length];                                                        //first parameter of any partial derivative[lstm]
        for(int l=0; l<wY.length; l++) {
            for(int c=0; c<wY[l].length; c++) {
                lMulWy[c]+= delLdelZ[l] * wY[l][c];
            }
        }
        delLdelWy= new float[delLdelZ.length][hiddenState.length];
        for(int l=0; l<delLdelWy.length; l++) {
            for(int c=0; c<hiddenState.length; c++) {
                delLdelWy[l][c]= delLdelZ[l] * hiddenState[c];
            }
        }
        //delLdelWy[][] is calculated

        delLdelWo= new float[outputGate.getResult().length][Xt.length];                                                 //first let's update the weights in the output gate
        float[]tanhCS= new float[cellState.length];
        for(int i=0; i< cellState.length; i++) {
            tanhCS[i]= (float)TANH.calc(cellState[i]);
        }
        float[] sigmoidDerivativeOfO= new float[outputGate.getResult().length];
        for(int i=0; i<sigmoidDerivativeOfO.length; i++) {
            sigmoidDerivativeOfO[i]= (float)SIGMOID_DERIVATIVE.calc(outputGate.getResult()[i]);
        }
        for(int l=0; l<delLdelWo.length; l++) {
            for(int c=0; c< Xt.length; c++) {
                delLdelWo[l][c] = (lMulWy[l]+ nextCell.dHSt()[l])* tanhCS[l]* sigmoidDerivativeOfO[l]* Xt[c];           //but the actual d l dhst is a bit more complicated than this D L dhst actually contains both the current change and the change from the previous step therefore DHS + one should be included in the calculation of d l DHS
            }
        }
        //delLdelWo[][] is calculated
        //BiasIncrement
/*        float[]sigmoidDerivativeOfOsBias= new float[]{  (float)SIGMOID_DERIVATIVE.calc(bO[0]),
                (float)SIGMOID_DERIVATIVE.calc(bO[1])};

        for(int i=0; i< delLbO.length; i++) {
            delLbO[i] = lMulWy[i] * outputGate.getResult()[i] * tanhCS[i] * sigmoidDerivativeOfOsBias[i] * xt[i];       //partielle Ableitung nach entspr. Bias
        }
        outputGate.setBias(delLbO);                                                                                     //Bias geupdated
        outputGate.calculateResult();                                                                                   //Auf Basis des geupdateten Output-Gate-Biases wird die Output-Gate-Ausgabe berechnet/aktualisiert
        //...BiasIncrement
*/
        delLdelWf= new float[forgetGate.getResult().length][Xt.length];
        float[]tanhDerivativeOfCS= new float[cellState.length];
        for(int i=0; i<tanhDerivativeOfCS.length; i++) {
            tanhDerivativeOfCS[i]= (float)TANH_DERIVATIVE.calc(cellState[i]);
        }
        float[]sigmoidDerivativeOfF= new float[]{   (float)SIGMOID_DERIVATIVE.calc(forgetGate.getResult()[0]),
                                                    (float)SIGMOID_DERIVATIVE.calc(forgetGate.getResult()[1])};
        for(int l=0; l<delLdelWf.length; l++) {
            for(int c=0; c< Xt.length; c++) {
                delLdelWf[l][c]= (lMulWy[l]* outputGate.getResult()[l]* tanhDerivativeOfCS[l]+ nextCell.dCSt()[l]) *    //of course DL D CST also needs to consider what is transmitted from the previous state anyway we have expanded d l d cstt so far
                                 prevCell.cellState[l]* sigmoidDerivativeOfF[l]* Xt[c];
            }
        }
        //delLdelWf[][] is calculated
        //BiasIncrement
/*        float[]sigmoidDerivativeOfFsBias= new float[]{  (float)SIGMOID_DERIVATIVE.calc(bF[0]),
                (float)SIGMOID_DERIVATIVE.calc(bF[1])};

        for(int i=0; i< delLbO.length; i++) {
            delLbF[i]= lMulWy[i] * outputGate.getResult()[i] * tanhDerivativeOfCS[i] * nextCell.dCSt()[i] *
                    sigmoidDerivativeOfFsBias[i] * xt[i];                                                               //partielle Ableitung nach entspr. Bias
        }
        forgetGate.setBias(delLbF);                                                                                     //Bias geupdated
        forgetGate.calculateResult();                                                                                   //Auf Basis des geupdateten Forget-Gate-Biases wird die Forget-Gate-Ausgabe berechnet/aktualisiert
        //...BiasIncrement
*/
        delLdelWi= new float[inputGate.getResult().length][Xt.length];
        float[]sigmoidDerivativeOfI= new float[]{   (float)SIGMOID_DERIVATIVE.calc(inputGate.getResult()[0]),
                (float)SIGMOID_DERIVATIVE.calc(inputGate.getResult()[1])};
        for(int l=0; l<delLdelWi.length; l++) {
            for(int c=0; c< Xt.length; c++) {
                delLdelWi[l][c]= lMulWy[l] * outputGate.getResult()[l] * tanhDerivativeOfCS[l] *
                                 candidateGate.getResult()[l] * sigmoidDerivativeOfI[l] * Xt[c];
            }
        }
        //delLdelWi[][] is calculated
        //BiasIncrement
/*        float[]sigmoidDerivativeOfIsBias= new float[]{  (float)SIGMOID_DERIVATIVE.calc(bI[0]),
                (float)SIGMOID_DERIVATIVE.calc(bI[1])};

        for(int i=0; i< delLbI.length; i++) {
            delLbI[i]= lMulWy[i] * outputGate.getResult()[i] * tanhDerivativeOfCS[i] * cellState[i] *
                    sigmoidDerivativeOfIsBias[i] * xt[i];                                                               //partielle Ableitung nach entspr. Bias
        }
        inputGate.setBias(delLbI);
        inputGate.calculateResult();
        //...BiasIncrement
*/
        delLdelWc= new float[candidateGate.getResult().length][Xt.length];
        float[]tanhDerivativeOfC= new float[candidateGate.getResult().length];
        for(int i=0; i<tanhDerivativeOfC.length; i++) {
            tanhDerivativeOfC[i]= (float)TANH_DERIVATIVE.calc(candidateGate.getResult()[i]);
        }
        for(int l=0; l<candidateGate.getResult().length; l++) {
            for(int c=0; c< Xt.length; c++) {
                delLdelWc[l][c]= lMulWy[l] * outputGate.getResult()[l] * tanhDerivativeOfCS[l] *
                                 inputGate.getResult()[l] * tanhDerivativeOfC[l] * Xt[c];
            }
        }
        //delLdelWc[][] is calculated
        //BiasIncrement
/*        float[]tanhDerivativeOfCsBias= new float[]{  (float)TANH_DERIVATIVE.calc(bC[0]),
                (float)TANH_DERIVATIVE.calc(bC[1])};

        for(int i=0; i< delLbC.length; i++) {
            delLbC[i]= lMulWy[i] * outputGate.getResult()[i] * tanhDerivativeOfCS[i] * inputGate.getResult()[i] *
                    tanhDerivativeOfCsBias[i] * xt[i];                                                                  //partielle Ableitung nach entspr. Bias
        }
        candidateGate.setBias(delLbC);                                                                                  //Bias geupdated
        candidateGate.calculateResult();                                                                                //Auf Basis des geupdateten Forget-Gate-Biases wird die Forget-Gate-Ausgabe berechnet/aktualisiert
        //...BiasIncrement
*/
    }

    /**
     * @param wF,wI,wC,wO,wY    weights to get updated */
    public void updatingWeights(float[][]wF, float[][]wI, float[][]wC, float[][]wO, float[][]wY) {
        for(int l=0; l< wF.length; l++) {
            for(int c=0; c< wF[l].length; c++) {
                wF[l][c]+= ALPHA*((-1)*delLdelWf[l][c]);
            }
        }
        for(int l=0; l< wI.length; l++) {
            for(int c=0; c< wI[l].length; c++) {
                wI[l][c]+= ALPHA*((-1)*delLdelWi[l][c]);
            }
        }
        for(int l=0; l< wO.length; l++) {
            for(int c=0; c< wO[l].length; c++) {
                wO[l][c]+= ALPHA*((-1)*delLdelWo[l][c]);
            }
        }
        for(int l=0; l< wC.length; l++) {
            for(int c=0; c< wC[l].length; c++) {
                wC[l][c]+= ALPHA*((-1)*delLdelWc[l][c]);
            }
        }
        for(int l=0; l< wY.length; l++) {
            for(int c=0; c< wY[l].length; c++) {
                wY[l][c]+= ALPHA*((-1)*delLdelWy[l][c]);
            }
        }
    }
}
