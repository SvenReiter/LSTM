package lstm2608010711_BerechneteResultateVonJedemTimestepSindWeggespeichert;

import java.util.ArrayList;

public class LstmCell {
    Func SIGM= x->(float) (1/(1+Math.exp(-x[0])));
    Func SIGM_DERIV= x->(SIGM.calc(x[0]) * (1-SIGM.calc(x[0])));
    Func TANH= x->(float) ((Math.exp(x[0])-Math.exp(-x[0])) / (Math.exp(x[0])+Math.exp(-x[0])));
    Func TANH_DERIV= x->(1.0 - (Math.pow(TANH.calc(x), 2)));
    float ETA= 0.1f;
    /* IMPORTANT:
        Der unterschied zwischen <LSTM> und <LstmCell> ist,
        dass <LstmCell> nur noch die Werte des aktuellen TimeSteps hält.
        DONE wie lässt sich das realisieren???
             ohne dass ich alle Werte pro TimeStep in Variablen/Felder halte???
        DONE indem Instanz-Werte gezielt überschrieben/als Redundanzen behalten werden

     */

    //Die Eingabe ist kein Bestandteil von <LstmCell> da sie von außen verabreicht wird(-> Im Konstruktor werden die Referenzen übergeben)
    float[] bf,bi,bc,bo;         //shared-weights
    float[][] wf,wi,wc,wo;       //shared-weights
    float[][] uf,ui,uc,uo;       //shared-weights

    ArrayList<float[]> hs, cs;               // TODO: 30.07.2026 die resultierende/berechneten Werte von jedem TimeStep müssen gehalten werden
    float[] hsPre;     //[Forwarding]Notwendig zur Berechnung des Netto-Gate-Werts bei der verrechnung des rekkurenten Gewichts.
                        //[BPTT-Berechnung der räumlichen Matrix-Gradienten]Notwendig zur Berechnung des aktuellen/neuen <delL_delU>.
    float[] csPre;     //[Forwarding]Notwendig zur Berechnung des aktuellen/neuen <cs>.
                        //[BPTT-Aufspaltung auf interne vektoren]Notwendig zur Berechnung des aktuellen/neuen <delF>.

    ArrayList<float[]> forgetGateNetto;              // DONE: 30.07.2026 die resultierende/berechneten Werte von jedem TimeStep müssen gehalten werden
    ArrayList<float[]> inputGateNetto;               // DONE: 30.07.2026 die resultierende/berechneten Werte von jedem TimeStep müssen gehalten werden
    ArrayList<float[]> candidateGateNetto;           // DONE: 30.07.2026 die resultierende/berechneten Werte von jedem TimeStep müssen gehalten werden
    ArrayList<float[]> outputGateNetto;              // DONE: 30.07.2026 die resultierende/berechneten Werte von jedem TimeStep müssen gehalten werden
    ArrayList<float[]> forgetGateActivation;         // DONE: 30.07.2026 die resultierende/berechneten Werte von jedem TimeStep müssen gehalten werden
    ArrayList<float[]> inputGateActivation;          // DONE: 30.07.2026 die resultierende/berechneten Werte von jedem TimeStep müssen gehalten werden
    ArrayList<float[]> candidateGateActivation;      // DONE: 30.07.2026 die resultierende/berechneten Werte von jedem TimeStep müssen gehalten werden
    ArrayList<float[]> outputGateActivation;         // DONE: 30.07.2026 die resultierende/berechneten Werte von jedem TimeStep müssen gehalten werden

    float[][] delL_delHs;
    float[] delL_delO;
    float[] delO;
    float[][] delL_delCs;
    float[][] delL_delCsTotal;
    float[] delC;
    float[] delI;
    float[] delF;

    float[][] delL_delWo;
    float[][] delL_delUo;
    float[][] delL_delWc;
    float[][] delL_delUc;
    float[][] delL_delWi;
    float[][] delL_delUi;
    float[][] delL_delWf;
    float[][] delL_delUf;

    float[][] delL_delWoTotal;
    float[][] delL_delWcTotal;
    float[][] delL_delWiTotal;
    float[][] delL_delWfTotal;

    float[][] delL_delUoTotal;
    float[][] delL_delUcTotal;
    float[][] delL_delUiTotal;
    float[][] delL_delUfTotal;

    float[] delL_delBoTotal;
    float[] delL_delBcTotal;
    float[] delL_delBiTotal;
    float[] delL_delBfTotal;

    public LstmCell(float[] bf, float[] bi, float[] bc, float[] bo,
                    float[][] wf, float[][] wi, float[][] wc, float[][] wo,
                    float[][] uf, float[][] ui, float[][] uc, float[][] uo, float[] target) {
        this.bf = bf; this.bi = bi; this.bc = bc; this.bo = bo;
        this.wf = wf; this.wi = wi; this.wc = wc; this.wo = wo;
        this.uf = uf; this.ui = ui; this.uc = uc; this.uo = uo;
        csPre= new float[bf.length];
        hsPre= new float[bf.length];
        cs= new ArrayList<>();
        hs= new ArrayList<>();
        forgetGateNetto= new ArrayList<>();
        inputGateNetto= new ArrayList<>();
        candidateGateNetto= new ArrayList<>();
        outputGateNetto= new ArrayList<>();
        forgetGateActivation= new ArrayList<>();
        inputGateActivation= new ArrayList<>();
        candidateGateActivation= new ArrayList<>();
        outputGateActivation= new ArrayList<>();
    }
    
    void bptt(float[]y, float[]...x) throws NumberFormatException{
        //Zeitschritt t=3
        int TIME_STEP= 2;   //:="Zeitschritt t=3"
        // DONE: 22.07.2026 Initialer Fehler am Ausgang
        if(y.length!= hs.get(TIME_STEP).length) throw new NumberFormatException("The targets length and HiddenStates length doesnt match!");
        delL_delHs= new float[TIME_STEP+1][hs.get(TIME_STEP).length];
        for(int i=0; i<delL_delHs[TIME_STEP].length; i++) {
            delL_delHs[TIME_STEP][i]= 2*(hs.get(TIME_STEP)[i]-y[i]);
        }
        /* DONE: 22.07.2026 Aufspaltung auf die internen
                            Vektoren(der Fehler am Ausgang wird partiell aufgespalten auf die interen Vektoren) */
        float[]tanh= new float[hs.get(TIME_STEP).length];
        for(int i=0; i<delL_delHs[TIME_STEP].length; i++) {
            tanh[i]= (float)TANH.calc(cs.get(TIME_STEP)[i]);
        }
        delL_delO= hadamardProd(delL_delHs[TIME_STEP], tanh);

        float[]delO= new float[hs.get(TIME_STEP).length];
        for(int i=0; i<hs.get(TIME_STEP).length; i++) {
            delO[i]= delL_delO[i]*(float)SIGM_DERIV.calc(outputGateActivation.get(TIME_STEP)[i]);
        }

        delL_delCs= new float[TIME_STEP+1][hs.get(TIME_STEP).length];
        delL_delCs[TIME_STEP]= hadamardProd(delL_delHs[TIME_STEP], outputGateActivation.get(TIME_STEP));
        for(int i=0; i<hs.get(TIME_STEP).length; i++) {     //for extending the Hadamard product to a third term in simple way
            delL_delCs[TIME_STEP][i]*= (float)TANH_DERIV.calc(cs.get(TIME_STEP)[i]);
        }

        delC= hadamardProd(delL_delCs[TIME_STEP], inputGateActivation.get(TIME_STEP-1));
        for(int i=0; i<hs.get(TIME_STEP).length; i++) {     //for extending the Hadamard product to a third term in simple way
            delC[i]*= (float)(1-Math.pow(candidateGateActivation.get(TIME_STEP-1)[i],2));
        }

        delI= hadamardProd(delL_delCs[TIME_STEP], candidateGateActivation.get(TIME_STEP-1));
        for(int i=0; i<hs.get(TIME_STEP).length; i++) {     //for extending the Hadamard product to a third term in simple way
            delI[i]*= (float)(SIGM_DERIV.calc(inputGateActivation.get(TIME_STEP-1)[i]));
        }

        delF= hadamardProd(delL_delCs[TIME_STEP], csPre);
        for(int i=0; i<hs.get(TIME_STEP).length; i++) {     //for extending the Hadamard product to a third term in simple way
            delF[i]*= (float)(SIGM_DERIV.calc(forgetGateActivation.get(TIME_STEP-1)[i]));
        }

        // DONE: 22.07.2026 Berechnung der räumlichen Matrix-Gradienten via äußeres Produkt (für t=3)
        delL_delWo= new float[outputGateActivation.get(TIME_STEP).length][x[x.length-1].length];
        for(int i=0; i<delL_delWo.length; i++) {
            for(int j=0; j<delL_delWo[0].length; j++) {
                delL_delWo[i][j]= delO[i]* x[x.length-1][i];
            }
        }
        delL_delUo= new float[outputGateActivation.get(TIME_STEP).length][x[x.length-1].length];
        for(int i=0; i<delL_delUo.length; i++) {
            for(int j=0; j<delL_delUo[0].length; j++) {
                delL_delUo[i][j]= delO[i]* hsPre[j];
            }
        }
        System.out.println();
        //Zeitschritt t=2
        // DONE: 29.07.2026 Transformiere delL_delHS[] zu delL_delHS[][] um alle HSs für das finale Gewichts-Update zugreifbar zu halten
        // DONE: 29.07.2026 Transformiere delL_delCS[] zu delL_delCS[][] um alle CSs zugreifbar zu halten
        // DONE: 22.07.2026 Fehlererweiterung durch die Zeit t=3 -> t=2
        TIME_STEP= 1;   //:="Zeitschritt t=2"
        delL_delHs[TIME_STEP]= vecAdd(vecAdd(matMul(uf, delF), matMul(ui, delI)), vecAdd(matMul(uc, delC), matMul(uo, delO)));
        delL_delCs[TIME_STEP]= hadamardProd(delL_delCs[TIME_STEP+1], forgetGateActivation.get(TIME_STEP-1));
            //IMPORTANT: for extending the Hadamard product to a third term in simple way wird hier nicht gebraucht
        // TODO: 22.07.2026 Aufspaltung auf die internen Vektoren
            // TODO: 29.07.2026 Transformiere delL_delCsTotal[] zu delL_delCsTotal[][] um alle CsTotals zugreifbar zu halten!!!
        // TODO: 30.07.2026 Über die Zeit{TIME_STEP} muss ich auch alle HS, CS und Aktivierungen der Gates zugreifbar halten!!! 
        delL_delCsTotal= new float[TIME_STEP+1][hs.get(TIME_STEP).length];
        float[]tanhDerivTmp= new float[hs.get(TIME_STEP).length];
        for(int i=0; i< tanhDerivTmp.length; i++) {
            tanhDerivTmp[i]= (float)TANH_DERIV.calc(cs.get(TIME_STEP)[TIME_STEP]);
        }
        delL_delCsTotal[TIME_STEP]= vecAdd(delL_delCs[TIME_STEP], hadamardProd(delL_delHs[TIME_STEP], hadamardProd(outputGateActivation.get(TIME_STEP), tanhDerivTmp)));
        // TODO: 22.07.2026 Räumlicher Matrix-Gradient ... hier wiederholen sich die Schritte langsam 
        // TODO: 22.07.2026 ...VERALGEMEINERUNG
    }

    void forwarding(int TIME_STEP, float[] x) {
        // DONE: 21.07.2026 Gate-Nettos berechnen
        forgetGateNetto.add(vecAdd(vecAdd(matMul(uf, hsPre), matMul(wf, x)), bf));
        inputGateNetto.add(vecAdd(vecAdd(matMul(ui, hsPre), matMul(wi, x)), bi));
        candidateGateNetto.add(vecAdd(vecAdd(matMul(uc, hsPre), matMul(wc, x)), bc));
        outputGateNetto.add(vecAdd(vecAdd(matMul(uo, hsPre), matMul(wo, x)), bo));

        // DONE: 21.07.2026 Anwendung der entspr. Aktivierungsfunktion auf entspr. Gate-Nettos
        float[]forgetGateActivationTMP= new float[x.length];
        float[]inputGateActivationTMP= new float[x.length];
        float[]candidateGateActivationTMP= new float[x.length];
        float[]outputGateActivationTMP= new float[x.length];
        forgetGateActivation.add(forgetGateActivationTMP);
        inputGateActivation.add(inputGateActivationTMP);
        candidateGateActivation.add(candidateGateActivationTMP);
        outputGateActivation.add(outputGateActivationTMP);
        for(int i=0; i<x.length; i++) {
            forgetGateActivationTMP[i]= (float)SIGM.calc(forgetGateNetto.get(forgetGateNetto.size()-1)[i]);
            inputGateActivationTMP[i]= (float)SIGM.calc(inputGateNetto.get(inputGateNetto.size()-1)[i]);
            candidateGateActivationTMP[i]= (float)TANH.calc(candidateGateNetto.get(candidateGateNetto.size()-1)[i]);
            outputGateActivationTMP[i]= (float)SIGM.calc(outputGateNetto.get(candidateGateNetto.size()-1)[i]);
        }
        inputGateActivation.set(inputGateActivation.size()-1, inputGateActivationTMP);                  //Die line ist glaube ich nicht nötig, weil die Referenz sowieso besteht, PRÜFEN!!!
        forgetGateActivation.set(forgetGateActivation.size()-1, forgetGateActivationTMP);               //Die line ist glaube ich nicht nötig, weil die Referenz sowieso besteht, PRÜFEN!!!
        candidateGateActivation.set(candidateGateActivation.size()-1, candidateGateActivationTMP);      //Die line ist glaube ich nicht nötig, weil die Referenz sowieso besteht, PRÜFEN!!!
        outputGateActivation.set(outputGateActivation.size()-1, outputGateActivationTMP);      //Die line ist glaube ich nicht nötig, weil die Referenz sowieso besteht, PRÜFEN!!!

        // DONE: 21.07.2026 die neuen cs/hs berechnen
        cs.add(TIME_STEP,vecAdd(hadamardProd(forgetGateActivation.get(forgetGateActivation.size()-1), csPre), hadamardProd(inputGateActivation.get(inputGateActivation.size()-1), candidateGateActivation.get(candidateGateActivation.size()-1))));
        hs.add(TIME_STEP, new float[x.length]);
        for(int i=0; i<x.length; i++) {
            hs.get(TIME_STEP)[i]= (float)(outputGateActivation.get(TIME_STEP)[i]* TANH.calc(cs.get(TIME_STEP)[i]));
        }
        for(int i=0; i<x.length; i++) {
            csPre[i]= cs.get(TIME_STEP)[i];
            hsPre[i]= hs.get(TIME_STEP)[i];
        }
    }

    private float[] hadamardProd(float[]vecA, float[]vecB) {
        float[]res= new float[vecA.length];
        for(int i=0; i<res.length; i++) {
            res[i]= vecA[i]*vecB[i];
        }
        return res;
    }

    private float[] vecAdd(float[] vecA, float[] vecB) {
        float[]res= new float[vecA.length];
        for(int i=0; i<res.length; i++) {
            res[i]= vecA[i]+vecB[i];
        }
        return res;
    }

    public float[] matMul(float[][] mat, float[] vec) {
        float[]res= new float[mat.length];
        /*done mat * vec
            mat[0][0]*vec[0]+ mat[0][1]*vec[1]
            mat[1][0]*vec[0]+ mat[1][1]*vec[1]
         */
        for(int l=0; l<mat.length; l++) {
            for(int c=0; c<mat[l].length; c++) {
                res[l]+= mat[l][c]* vec[c];
            }
        }
        return res;
    }
}
