package lstm2604221130;

public class LSTM_cell {
    LSTM_cell prevCell, nextCell;
    final Func SIGMOID= x-> (float)(1 / (1 + Math.exp(-x[0])));
    //    final Func SIGMOID_DERIVATIVE= x->(SIGMOID.calc(x)*(1-SIGMOID.calc(x)));
    final Func SIGMOID_DERIVATIVE= x->(x[0]*(1-x[0]));
    final Func TANH= x-> (float)((Math.exp(x[0])-Math.exp(-x[0])) / (Math.exp(x[0])+Math.exp(-x[0])));
    final Func TANH_DERIVATIVE= x->(1.0 - (TANH.calc(x)*TANH.calc(x)));
    final Gate forgetGate;                                                                                              //Entscheidet, was vergessen wird.
    final Gate inputGate;                                                                                               //Entscheidet, welche neuen Informationen gespeichert werden.
    final Gate candidateGate;                                                                                           //Erstellt neue Kandidatenwerte.
    final Gate outputGate;                                                                                              //Entscheidet, was ausgegeben wird
    final float ALPHA= 0.000001f;                                                                                       //Lernrate
    final float ETA= 0.1f;
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
     * @param bF,bI,bC,bO bias-matrices
     * @param u rekurrente Gewichte und müssen der Dimensionierung des HiddenStates entspr.
     * @param concatXt (dirtyHack)
     * @param mulMitGetrimmtemInput (dirtyHack)dieser Flag wird weitergereicht bis zu Gate().calculateResult,
     *                               damit wird der Multiplikation mitgeteilt,
     *                               ob der input oder eine konkatenierte Version von input multipliziert werden soll*/
    public LSTM_cell(float[]preCs, float[]preHs, float[]xt, float[]y, float[][]wF, float[][]wI, float[][]wC, float[][]wO,
                     float[]bF, float[]bI, float[]bC, float[]bO, float u, boolean concatXt, boolean mulMitGetrimmtemInput,
                     boolean concatXtPrev) {
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

        if(concatXt) {
            Xt= new float[preHs.length+ xt.length];
            if(concatXtPrev) {
                for (int i = 0; i < xt.length; i++) {
                    Xt[i] = xt[i];                           //nach Xt[] werden zuerst die input-Werte aus xt[] geschrieben...
                }
                for (int i = 0; i < preHs.length; i++) {
                    Xt[xt.length + i] = preHs[i];              //...danach die Werte aus preHs[].
                }
            }else {
                for(int i=0; i<preHs.length; i++) {
                    Xt[i]=preHs[i];                         //nach Xt[] werden zuerst die Werte aus preHs[] geschrieben...
                }
                for(int i=0; i<xt.length; i++) {
                    Xt[preHs.length+i]=xt[i];               //...danach die input-Werte aus xt[].
                }
            }
        }else {
            Xt = new float[xt.length];
            for (int i = 0; i < xt.length; i++) {
                Xt[i] = xt[i];
            }
        }

        forgetGate= new Gate(SIGMOID, Xt, wF, bF, u, preHs, mulMitGetrimmtemInput);
        inputGate= new Gate(SIGMOID, Xt, wI, bI, u, preHs, mulMitGetrimmtemInput);
        candidateGate= new Gate(TANH, Xt, wC, bC, u, preHs, mulMitGetrimmtemInput);
        outputGate= new Gate(SIGMOID, Xt, wO, bO, u, preHs, mulMitGetrimmtemInput);
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
     * @param wY size of output/logit
     * @post yHat contains output/logit
     * @post <hiddenState> and <cellState> modified */
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
     * @post <hiddenState> and <cellState> modified */
    public void forward() {
        // sowohl cellState[] als auch hiddenState[] müssen vom Resultat der vorherigen Zelle übernommen werden
//        hiddenState= prevCell.dHSt();
//        cellState= prevCell.dCSt();
        for(int g=0; g<forgetGate.getResult().length; g++) {
            for (int i = 0; i < cellState.length; i++) {
                cellState[i] *= forgetGate.getResult()[g];
            }
            for(int i=0; i<cellState.length; i++) {
                cellState[i]+=inputGate.getResult()[g] * candidateGate.getResult()[g];
            }
            for(int i=0; i<hiddenState.length; i++) {
                hiddenState[i]= (float) (TANH.calc(cellState[i]) * outputGate.getResult()[g]);
            }
        }
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

    /**
     *
     * @param xt
     * @param wF, wI, wC, wO
     * @param b
     * @param delLdelHt         der Fehler am Ausgang
     * @param activeTest    um den Testfall von außen kontrollieren zu können
     */
    public void backpropagateGamma(float[] xt, float[][]wF, float[][]wI, float[][]wC, float[][]wO,
                                  float[] b, float delLdelHt, int activeTest) {
        // TODO: 17.04.2026
        //1. Gradient des Output Gates (delO)
        //Zuerst berechnen wir, wie stark der Fehler vom aktuellen Hidden State(h[t]) abhängt:
        float[] delEdelHt= new float[cellState.length];
        for(int i=0; i<cellState.length; i++) {
            delEdelHt[i]= hiddenState[i]-delLdelHt;
        }
        float[] delLdelO= new float[cellState.length];
        float[] delO= new float[cellState.length];
        for(int i=0; i<cellState.length; i++) {
            delLdelO[i]= delEdelHt[i] * (float)TANH.calc(cellState[i]);
            //Der Gradient vor der Aktivierungsfunktion (Sigmoid) ist:
            delO[i]= delLdelO[i]* (float)SIGMOID_DERIVATIVE.calc(outputGate.getResult()[0]);
        }
        for(int i=0; i<wO[0].length; i++) {
            wO[0][i]-= delO[i]* xt[i]* ETA;
        }
        //2. Gradient des Cell States (delC)
        //Der Fehler fließt zurück in den Cell State:
        float[] delLdelC= new float[cellState.length];
        for(int i=0; i<delLdelC.length; i++) {
            delLdelC[i]= delEdelHt[i]* outputGate.getResult()[0]*(1-(float)Math.pow(TANH.calc(cellState[i]),2));
        }
        //3. Gradienten der internen Komponenten (delI, delCellCandidate, delF)
        //Input Gate:
        float[] delI= new float[cellState.length];
        for(int i=0; i<delI.length; i++) {
            delI[i]= delLdelC[i]* candidateGate.getResult()[0]* (float)SIGMOID_DERIVATIVE.calc(inputGate.getResult()[0]);
        }
        for(int i=0; i<wI[0].length; i++) {
            wI[0][i]-= delI[i]* xt[i]* ETA;
        }
        //Cell Candidate:
        float[] delC= new float[cellState.length];
        for(int i=0; i<delC.length; i++) {
            delC[i]= delLdelC[i]* inputGate.getResult()[0]* (1-(float)Math.pow(TANH.calc(candidateGate.getResult()[0]),2));
        }
        for(int i=0; i<wC[0].length; i++) {
            wC[0][i]-= delC[i]* xt[i]* ETA;         //Grundsätzlich <GewichtUpdate>:= <Gradient>*<Input>*<ETA>
        }
        //Forget Gate:
        float[] delLdelF= new float[cellState.length];
        float[] delFNet= new float[cellState.length];
        if(prevCell.cellState[0]!=0) {  /*Wenn c[t-1]=0 => das Forget Gate in diesem ersten Schritt keinen Einfluss auf das Ergebnis
         *  Das wurde in AnotherTest_LSTM_cell().testsBackpropagation() definiert
         *   Definition von LSTM_cell prevNode, 1. Argument*/
            //1. Der Einfluss des Forget Gates auf den Cell State
            for(int i=0; i<delLdelF.length; i++) {
                delLdelF[i] = delLdelC[i] * prevCell.cellState[0];
            }
            //2. Gradient vor der Aktivierungsfunktion(delFNet[])
            for(int i=0; i<delFNet.length; i++) {
                delFNet[i]= delLdelF[i]* (float)SIGMOID_DERIVATIVE.calc(forgetGate.getResult()[0]);
            }
            //3. Update der Gewichte für das Forget Gate(wF[][])
            //Die Anpassung der Gewichte berechnet sich wieder durch die Multiplikation mit dem Input
            for(int i=0; i<wF[0].length; i++) {
                wF[0][i]-= delFNet[i]* xt[i]* ETA;
            }
            // DONE: 20.04.2026 ForgetGate-Bias updaten
            float[] bF_neu= new float[bF.length];
            for( int i=0; i< bF.length; i++) {
                bF_neu[i]= bF[i]- ETA* delFNet[i];
            }
        }
        /* DONE: 20.04.2026 Werte vergleichen -> Berechnung der Bias-Updates
            delO
            delI
            delF
         */
        float[] bI_neu= new float[bI.length];
        for( int i=0; i< bI.length; i++) {
            bI_neu[i]= bI[i]- ETA* delI[i];
        }
        float[] bO_neu= new float[bO.length];
        for( int i=0; i< bO.length; i++) {
            bO_neu[i]= bO[i]- ETA* delO[i];
        }
        float[] delUF= new float[prevCell.hiddenState.length];
        float[] delUI= new float[prevCell.hiddenState.length];
        float[] delUC= new float[prevCell.hiddenState.length];
        float[] delUO= new float[prevCell.hiddenState.length];
        for(int i=0; i< prevCell.hiddenState.length; i++) {
            delUF[i]= delFNet[i]* prevCell.hiddenState[i];
            delUI[i]= delI[i]* prevCell.hiddenState[i];
            delUC[i]= delC[i]* prevCell.hiddenState[i];
            delUO[i]= delO[i]* prevCell.hiddenState[i];
        }

        switch(activeTest) {
            case 1: {
                AnotherTest_LSTM_cell.toTestA(delLdelO, delO, wO, delLdelC, delC, wC, delI, wI, delLdelF, delFNet, wF, delUF, delUI, delUC, delUO);
                break;
            }
            case 2: {
                AnotherTest_LSTM_cell.toTestB(delO, wO, delC, wC, delI, wI, delFNet, wF, delLdelC, delUF, delUI, delUC, delUO);
                break;
            }
        }
    }

    /**
     *
     * @param xt
     * @param wF, wI, wC, wO
     * @param b
     * @param delLdelHt         der Fehler am Ausgang
     * @param activeTest    um den Testfall von außen kontrollieren zu können
     */
    public void backpropagateBeta(float[] xt, float[][]wF, float[][]wI, float[][]wC, float[][]wO,
                                  float[] b, float delLdelHt, int activeTest) {
        // TODO: 17.04.2026
    //1. Gradient des Output Gates (delO)
      //Zuerst berechnen wir, wie stark der Output-Zustand das Ergebnis beeinflusst:
        float[] delLdelO= new float[cellState.length];
        float[] delO= new float[cellState.length];
        for(int i=0; i<cellState.length; i++) {
            delLdelO[i]= delLdelHt * (float)TANH.calc(cellState[i]);
          //Der Gradient vor der Aktivierungsfunktion (Sigmoid) ist:
            delO[i]= delLdelO[i]* (float)SIGMOID_DERIVATIVE.calc(outputGate.getResult()[0]);
        }
        for(int i=0; i<wO[0].length; i++) {
            wO[0][i]-= delO[i]* xt[i]* ETA;
        }
    //2. Gradient des Cell States (delC)
      //Der Fehler fließt zurück in den Cell State:
        float[] delLdelC= new float[cellState.length];
        for(int i=0; i<delLdelC.length; i++) {
            delLdelC[i]= delLdelHt* outputGate.getResult()[0]*(1-(float)Math.pow(TANH.calc(cellState[i]),2));
        }
    //3. Gradienten der internen Komponenten (delI, delCellCandidate, delF)
      //Input Gate:
        float[] delI= new float[cellState.length];
        for(int i=0; i<delI.length; i++) {
            delI[i]= delLdelC[i]* candidateGate.getResult()[0]* (float)SIGMOID_DERIVATIVE.calc(inputGate.getResult()[0]);
        }
        for(int i=0; i<wI[0].length; i++) {
            wI[0][i]-= delI[i]* xt[i]* ETA;
        }
      //Cell Candidate:
        float[] delC= new float[cellState.length];
        for(int i=0; i<delC.length; i++) {
            delC[i]= delLdelC[i]* inputGate.getResult()[0]* (1-(float)Math.pow(TANH.calc(candidateGate.getResult()[0]),2));
        }
        for(int i=0; i<wC[0].length; i++) {
            wC[0][i]-= delC[i]* xt[i]* ETA;         //Grundsätzlich <GewichtUpdate>:= <Gradient>*<Input>*<ETA>
        }
      //Forget Gate:
        float[] delLdelF= new float[cellState.length];
        float[] delFNet= new float[cellState.length];
        if(prevCell.cellState[0]!=0) {  /*Wenn c[t-1]=0 => das Forget Gate in diesem ersten Schritt keinen Einfluss auf das Ergebnis
                                         *  Das wurde in AnotherTest_LSTM_cell().testsBackpropagation() definiert
                                         *   Definition von LSTM_cell prevNode, 1. Argument*/
            //1. Der Einfluss des Forget Gates auf den Cell State
            for(int i=0; i<delLdelF.length; i++) {
                delLdelF[i] = delLdelC[i] * prevCell.cellState[0];
            }
            //2. Gradient vor der Aktivierungsfunktion(delFNet[])
            for(int i=0; i<delFNet.length; i++) {
                delFNet[i]= delLdelF[i]* (float)SIGMOID_DERIVATIVE.calc(forgetGate.getResult()[0]);
            }
            //3. Update der Gewichte für das Forget Gate(wF[][])
            //Die Anpassung der Gewichte berechnet sich wieder durch die Multiplikation mit dem Input
            for(int i=0; i<wF[0].length; i++) {
                wF[0][i]-= delFNet[i]* xt[i]* ETA;
            }
        }

        // TODO: 17.04.2026 Berechnung der Gradienten für <uF>, <uI>, <uC>, <uO>
        float[] delUF= new float[prevCell.hiddenState.length];
        float[] delUI= new float[prevCell.hiddenState.length];
        float[] delUC= new float[prevCell.hiddenState.length];
        float[] delUO= new float[prevCell.hiddenState.length];
        for(int i=0; i< prevCell.hiddenState.length; i++) {
            delUF[i]= delFNet[i]* prevCell.hiddenState[i];
            delUI[i]= delI[i]* prevCell.hiddenState[i];
            delUC[i]= delC[i]* prevCell.hiddenState[i];
            delUO[i]= delO[i]* prevCell.hiddenState[i];
        }

        switch(activeTest) {
            case 1: {
                AnotherTest_LSTM_cell.toTestA(delLdelO, delO, wO, delLdelC, delC, wC, delI, wI, delLdelF, delFNet, wF, delUF, delUI, delUC, delUO);
                break;
            }
            case 2: {
                AnotherTest_LSTM_cell.toTestB(delO, wO, delC, wC, delI, wI, delFNet, wF, delLdelC, delUF, delUI, delUC, delUO);
                break;
            }
        }
    }
}
