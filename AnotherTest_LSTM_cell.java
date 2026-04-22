package lstm2604221130;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class AnotherTest_LSTM_cell {

    @Test
    public void testsUWeights() {
        /*TODO first implement static calculations of
        Was ist das?
            delUF: Gradient für die Recurrent Weights des Forget Gates.
            delUI: Gradient für die Recurrent Weights des Input Gates.
            delUC: Gradient für die Recurrent Weights des Cell Candidates.
            delUO: Gradient für die Recurrent Weights des Output Gates.
            In der Backpropagation-Logik eines LSTM stehen diese Variablen
            für die akkumulierten Gradienten der Recurrent Weights (U).
        Wozu braucht man das?
            Während w[][] die Gewichte für den aktuellen Input (x[t])sind,
            verknüpfen die Gewichte U[] den versteckten Zustand des vorherigen
            Zeitschritts (h[t-1]) mit den Gates.
        Warum sind sie wichtig?
          1)	Backpropagation Through Time (BPTT): Da ein LSTM über Sequenzen
                lernt, wird der Fehler nicht nur auf die aktuellen Input-
                Gewichte (w) zurückgeführt, sondern auch darauf, wie das
                Gedächtnis (h[t-1]) genutzt wurde.
          2)	Akkumulation: In der Praxis berechnet man diese Gradienten über
                eine ganze Sequenz hinweg.
                Man addiert die Fehleränderungen pro Zeitschritt in diesen
                delU-Variablen auf, bevor man am Ende der Sequenz ein einziges
                Mal die Gewichte U updatet.
          3)	Berechnung: Im obigen Beispiel wäre die Berechnung für
                delUI(Input Gate) zum Beispiel:
                 Gradient des Gates(delO/F/C/I)* vorheriger HiddenState(h[t-1])
                h[t-1]			:= 0.046
                Aktivierung f[t]:= 0.68
                Aktivierung i[t]:= 0.539
                Aktivierung C[t]:= 0.17
                delForget		:= -0.0041
                delInput		:= -0.0099
                delCandidate	:= -0.1229
                delOutput		:= -0.0147
                \
                 =>	delUF:= -0.0041* 0.046= -0.00019
                    delUI:= -0.0099* 0.046= -0.00046
                    delUC:= -0.1229* 0.046= -0.00565
                    delUO:= -0.0147* 0.046= -0.00068
        Wo wird es verrechnet? Bzw. wo wird dieser Wert genutzt/verwendet?
            1. Forget Gate
              Formel: 		σ(WF*x[t+1]		   +UF*h[t]+b[f])
              Rechnung:	σ((0.5*1)+(0.5*0.5)+(0.1*0.046)+0)
            2. Input Gate
              Formel: 		σ(WI*x[t]+UI*h[t]+b[i])
              Rechnung:		σ((0.1*1)+(0.1*0.5)+(0.1*0.046)+0)
            3. CellCandidate
              Formel: 	 tanh(WC*x[t]+UC*h[t]+b[c])
              Rechnung:	 tanh((0.1*1)+(0.1*0.5)+(0.1*0.046)+0)
                        =tanh(0.15)= 0.149
            4. Neuer CellState(Hier fließt auch der alte Cell State ein)
              Formel: 	 f[t+1]*c[t]+i[t+1]*Candidate[t+1]
              Rechnung:	 0.68*0.08  +0.539*0.153
            5. Output Gate
              Formel: 	σ(WO*x[t]+UO*h[t]+b[o])
              Rechnung:	σ((0.2*1)+(0.2*0.5)+(0.1*0.046)+0)
              HiddenState: OutputGate* tanh(Neuer CellState)
        Wo wird es angepasst? Bzw. wo wird dieser Wert angepasst?
            Nach Backpropagation
                U[neu]= U[alt]-(Lernrate*Gradient)
        Wie berechnet man die Gradienten der rekurrenten Gewichte?
            Der Gradient für das Recurrent Weight ergibt sich aus
            dem Fehler am Gate und dem vorherigen Hidden State
            delUO berechnen:
            delUO= delO* h[t-1]
                 = 0.0098* 0.1= 0.00098         //Lernrate:= 0.1
            delUO[neu]= 0.1-(0.1*0.00098)
            .
            delUF berechnen:
                c[t-1]:= 0.4
                f[t]:= 0.679
                h[t-1]:= 0.1
                Fehler am HiddenState(delH[t]):= 0.5
                Lernrate:= 0.1
                SchrittA: Fehler am CellState(delC[t])
                    delC[t]= delH[t]* o[t] * (1-tanh²(c[t]))
                           = 0.5    * 0.574* (1-0.08²)
                           = 0.285
                SchrittB: Fehler am Forget Gate(delF[t])
                    σ'(z[f])= f[t]*(1-f[t])= 0.679*(1-0.679)= 0.218
                    delF[t]= delC[t]* c[t-1]* σ'(z[f])
                           = 0.285* 0.4* 0.218
                           = 0.0248
                Schritt C: Der Gradient(delU[f])
                    delUF= delF[t]*h[t-1]
                         = 0.0248* 0.1
                         = 0.00248
                Schritt D: Gewicht anpassen
                    UF_neu= UF_alt-(Lernrate*delUF)
                          = 0.1   -(0.1     *0.00248)
                          = 0.099752
            .
            delUI berechnen:
                delC[t]:= 0.285
                CellCandidate:= 0.149
                i[t]:= 0.537
                h[t-1]:= 0.1
                SchrittA: Ableitung der Aktivierungsfunktion
                    σ'(z[i])= i[t]* (1-i[t])
                            = 0.537* (1-0.537)
                            = 0.2486
                Schritt B: Fehler am Gate(delI[t])
                    delI[t]= delC[t]* CellCandidate* σ'(z[i])
                           = 0.285  * 0.149        * 0.2486
                           = 0.01056
                Schritt C: Der Gradient(delU[i])
                    delUI= delI[t]*h[t-1]
                         = 0.01056* 0.1
                         = 0.001056
                Schritt D: Gewicht anpassen
                    UI_neu= UI_alt-(Lernrate*delUI)
                          = 0.1   -(0.1     *0.001056)
                          = 0.0998944
            .
            delUC berechnen:
                delC[t]:= 0.285
                i[t]:= 0.537
                CellCandidate:= 0.149
                h[t-1]:= 0.1
                Lernrate:= 0.1
                Schritt A: Ableitung der Aktivierungsfunktion(tanh')
                    tanh'(z[c])= 1-(CellCandidate[t])²
                    tanh'(z[c])= tanh'(CellCandidate)= tanh'(0.15)
                    tanh'(0.15)= 1-(0.149)²
                               = 1-0.0222
                               = 0.9778
                Schritt B: Fehler am Candidate-Gate(delCellCandidate[t])
                    delCellCandidate[t]= delC[t]* i[t]*  tanh'(z[c])
                                       = 0.285  * 0.537* 0.9778
                                       = 0.1496
                Schritt C: Der Gradient(delUC)
                    delUC= delCellCandidate[t]* h[t-1]
                         = 0.1496* 0.1
                Schritt D: Gewicht anpassen
                    UC_neu= UC_alt-(Lernrate*delUC)
                          = 0.1   -(0.1     *0.001496)
                          = 0.098504
            .
            delUO berechnen:
                delH[t]:= 0.5
                c[t]:= 0.080
                0[t]:= 0.574
                h[t-1]:= 0.1
                Lernrate:= 0.1
                Schritt A: Ableitung der Aktivierungsfunktion(σ')
                    σ'(z[o])= o[t]*(1-o[t])
                            = 0.574*(1-0.574)
                            = 0.2445
                Schritt B: Fehler am Gate(delO[t])
                    delO[t]= delH[t]* tanh(c[t])* σ'(z[o])
                           = 0.5* 0.0798* 0.2445
                           = 0.00976
                Schritt C: Der Gradient(delU[o])
                    delUO= delO[t]* h[t-1]
                           = 0.00976* 0.1
                           = 0.000976
                Schritt D: Gewicht anpassen
                    UO_neu= UO_alt-(Lernrate*delUO)
                          = 0.1   -(0.1     *0.000976)
                          = 0.0999024
     */
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.5f, 0.5f}};
        float[][]wI= new float[][]{{0.1f, 0.1f}};
        float[][]wC= new float[][]{{0.1f, 0.1f}};
        float[][]wO= new float[][]{{0.2f, 0.2f}};
        float u= 0.1f;
        LSTM_cell node= new LSTM_cell(
                new float[]{0.f, 0.f, 0.f},      //Um Rechnung überschaubar zu halten, setzen wir die Größe des
                new float[]{0.f, 0.f, 0.f},      //  Hidden-/Cell State (Anzahl der Neuronen in der LSTM-Zelle) auf 1.
                new float[]{1f, 0.5f},   //(Ein 2D-Vektor)
                new float[]{0,1,0},
                wF, wI, wC, wO,
                new float[]{0.0f}, new float[]{0.0f},
                new float[]{0.0f}, new float[]{0.0f},
                u, false, false, false);
        LSTM_cell nextNode= new LSTM_cell(new float[]{1f,0f}, new float[]{0f,0f},
                new float[]{1f,0f,0f}, new float[]{0,1,0}, wF, wI, wC, wO,
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f}, 0.1f, true, false, false);

        LSTM_cell prevNode= new LSTM_cell(new float[]{0.4f}, new float[]{0.1f},
                new float[]{1f,0f,0f}, new float[]{0,1,0}, wF, wI, wC, wO,
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f}, 0.1f, true, false, false);
        node.nextCell = nextNode;
        node.prevCell = prevNode;   //Um CS,HS der vorherigen Celle zu definieren, muss <prevNode> geändert werden
        node.forward();
        float aktuellerFehlerAmHS_t= 0.5f;  //delHt
        final Func SIGMOID= x-> (float)(1 / (1 + Math.exp(-x[0])));
        final Func SIGMOID_DERIVATIVE= x->((x[0])*(1-x[0]));
        final Func TANH= x-> (float)((Math.exp(x[0])-Math.exp(-x[0])) / (Math.exp(x[0])+Math.exp(-x[0])));
      //1. Fehler am Output Gate (del_o) berechnen
        float[] outputGatesDerivedMathMulResult= new float[node.outputGate.getZ().length];
        for(int i=0; i<outputGatesDerivedMathMulResult.length; i++) {
            outputGatesDerivedMathMulResult[i]= (float)SIGMOID_DERIVATIVE.calc(node.outputGate.getResult()[i]);
        }
        float delO= aktuellerFehlerAmHS_t* (float)TANH.calc(node.cellState[0])* (float)SIGMOID_DERIVATIVE.calc(SIGMOID.calc(node.outputGate.getZ()[0]));    //TODO delO evt als Vektor/Array
        assertEquals(0.0098, delO, DELTA);
      //2. Gradient für U_o berechnen
        float delU_o= delO* node.prevCell.hiddenState[0];           //TODO delU_o evt als Vektor/Array
        assertEquals(0.00098, delU_o, DELTA);
        final float LERNRATE= 0.1f;
        float U_o= 0.1f, U_f= 0.1f, U_i= 0.1f, U_c= 0.1f;
        float U_o_neu= U_o- (LERNRATE* delU_o);
        assertEquals(0.099902, U_o_neu, DELTA);

        //Schritt A: Fehler am Cell State (delC_t)
        float delH_t= 0.5f; //Fehler am aktuellen HiddenState
        float delC_t= delH_t* node.outputGate.getResult()[0]* (float)(1- Math.pow(TANH.calc(node.cellState[0]), 2));    //TODO delH_t evt als Vektor/Array
        assertEquals(0.285, delC_t, DELTA);

        //Schritt B: Fehler am Forget Gate (delF_t)
        float delF_t= delC_t* node.prevCell.cellState[0]* (float)SIGMOID_DERIVATIVE.calc(SIGMOID.calc(node.forgetGate.getZ()[0]));  //TODO delC_t evt als Vektor/Array
        assertEquals(0.0248, delF_t, DELTA);

      //3. Gradient und Update für  U_f
        //Schritt C: Der Gradient (delU_f)
        float delU_f= delF_t* node.prevCell.hiddenState[0];     //TODO delU_f evt als Vektor/Array
        assertEquals(0.00248, delU_f, DELTA);
        //Schritt D: Gewicht anpassen
        float U_f_neu= U_f - (LERNRATE* delU_f);
        assertEquals(0.099752, U_f_neu, DELTA);

      //4. Den Fehler zum Input Gate leiten (delI_t)
        //Schritt A: Ableitung der Aktivierungsfunktion
        float[] inputGatesDerivedMathMulResult= new float[node.inputGate.getZ().length];
        for(int i=0; i<inputGatesDerivedMathMulResult.length; i++) {
            inputGatesDerivedMathMulResult[i]= (float)SIGMOID_DERIVATIVE.calc(node.inputGate.getResult()[i]);
        }
        //Schritt B: Fehler am Gate (delI_t)
        float delI_t= delC_t* node.candidateGate.getResult()[0]* (float)SIGMOID_DERIVATIVE.calc(SIGMOID.calc(node.inputGate.getZ()[0]));
        System.out.println();
      //3. Gradient und Update für  U_i
        //Schritt C: Der Gradient (delU_i)
        float delU_i= delI_t* node.prevCell.hiddenState[0];     //TODO delU_i evt als Vektor/Array
        assertEquals(0.001056, delU_i, DELTA);
        //Schritt D: Gewicht anpassen
        float U_i_neu= U_i - (LERNRATE* delU_i);
        assertEquals(0.0998944, U_i_neu, DELTA);

      //5. Den Fehler zum Candidate Gate leiten (delC_t)
        //Schritt A: Ableitung der Aktivierungsfunktion (tanh')
        float[] candidateGatesDerivedMathMulResult= new float[node.candidateGate.getZ().length];
        for(int i=0; i<candidateGatesDerivedMathMulResult.length; i++) {
            candidateGatesDerivedMathMulResult[i]= (float)(1- Math.pow(node.candidateGate.getResult()[0], 2));
        }
        assertEquals(0.9778, candidateGatesDerivedMathMulResult[0], DELTA);
        //Schritt B: Fehler am Gate (delC_t)
        float FehlerDelC_t= delC_t* node.inputGate.getResult()[0]* (float)(1- Math.pow(node.candidateGate.getResult()[0], 2));
        assertEquals(0.1496, FehlerDelC_t, DELTA);

      //6. Gradient und Update für U_i
        //Schritt C: Der Gradient (delU_c)
        float delU_c= FehlerDelC_t* node.prevCell.hiddenState[0];     //TODO delU_c evt als Vektor/Array
        assertEquals(0.01496, delU_c, DELTA);
        //Schritt D: Gewicht anpassen
        float U_c_neu= U_c - (LERNRATE* delU_c);
        assertEquals(0.098504, U_c_neu, DELTA);

      //Vervollständigte Berechnung für rekurrentes Gewicht am Output Gate
        //Schritt A: Ableitung der Aktivierungsfunktion (sigmoid')
        for(int i=0; i<candidateGatesDerivedMathMulResult.length; i++) {
            candidateGatesDerivedMathMulResult[i]= node.outputGate.getResult()[0]* (1- node.outputGate.getResult()[0]);
        }
        assertEquals(0.2445, candidateGatesDerivedMathMulResult[0], DELTA);

        //Schritt B: Fehler am Gate (delO_t)
        float FehlerDelO_t= delH_t* (float)TANH.calc(node.cellState[0])* (float)(SIGMOID_DERIVATIVE.calc(node.outputGate.getResult()[0]));
        assertEquals(0.00976, FehlerDelO_t, DELTA);

        //Gradient und Update für U_o
        //Schritt C: Der Gradient (delU_o)
        delU_o= FehlerDelO_t* node.prevCell.hiddenState[0];     //TODO delU_c evt als Vektor/Array
        assertEquals(0.000976, delU_o, DELTA);
        //Schritt D: Gewicht anpassen
        U_o_neu= U_o - (LERNRATE* delU_o);
        assertEquals(0.0999024, U_o_neu, DELTA);

        /* TODO: 22.04.2026 Manchmal verwendet ich hier SIGMOID_DERIVATIVE.calc(SIGMOID.calc(x))
                            und manchmal bloß SIGMOID_DERIVATIVE.calc(x)
                            Welcher Aufruf liefert mir die Ableitung von sigmoid()
         */
    }
    
    @Test
    public void testsBackpropagationAdjustingBiases() {
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.5f, 0.5f}};
        float[][]wI= new float[][]{{0.1f, 0.1f}};
        float[][]wC= new float[][]{{0.1f, 0.1f}};
        float[][]wO= new float[][]{{0.2f, 0.2f}};
        float u= 0.1f;
        LSTM_cell node= new LSTM_cell(
                new float[]{0.f, 0.f, 0.f},      //Um Rechnung überschaubar zu halten, setzen wir die Größe des
                new float[]{0.f, 0.f, 0.f},      //  Hidden-/Cell State (Anzahl der Neuronen in der LSTM-Zelle) auf 1.
                new float[]{1f, 0.5f},   //(Ein 2D-Vektor)
                new float[]{0,1,0},
                wF, wI, wC, wO,
                new float[]{0.0f}, new float[]{0.0f},
                new float[]{0.0f}, new float[]{0.0f},
                u, false, false, false);
        LSTM_cell nextNode= new LSTM_cell(new float[]{1f,0f}, new float[]{0f,0f},
                new float[]{1f,0f,0f}, new float[]{0,1,0}, wF, wI, wC, wO,
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f}, 0.1f, true, false, false);

        LSTM_cell prevNode= new LSTM_cell(new float[]{0f}, new float[]{0f},
                new float[]{1f,0f,0f}, new float[]{0,1,0}, wF, wI, wC, wO,
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f}, 0.1f, true, false, false);
        node.nextCell = nextNode;
        node.prevCell = prevNode;
      //1. Die Gate-Berechnungen
        assertEquals(node.forgetGate.getResult()[0], 0.6792f, DELTA);
        assertEquals(node.inputGate.getResult()[0], 0.5374f, DELTA);
        assertEquals(node.outputGate.getResult()[0], 0.5744f, DELTA);
      //2. Der neue Zellzustand-Kandidat (~C[t])
        assertEquals(node.candidateGate.getResult()[0], 0.149f, DELTA);
        node.forward();
      //3. Update des Cell State (C[t])
        assertEquals(node.cellState[0], 0.080f, DELTA);
      //4. Output / Hidden State (h[t])
        assertEquals(node.hiddenState[0], 0.046f, DELTA);
      //Der Fehler am Output (del_h[t])
        float y= 0.5f;
        node.backpropagateGamma(node.xt, wF, wI, wC, wO,  new float[]{0f}, y, 2);
        // TODO: 22.04.2026 Auf unten im Kommentar beschriebenen Ergebnisse testen 
        /*
        Hier ist ein konkretes Rechenbeispiel für einen LSTM-Zeitschritt.
        Da du Gewichte paarweise (w= {0.5, 0.5}) und einen Input
        angegeben hast, berechnen wir die Gates basierend auf der Summe der gewichteten Inputs ({1, 0.5}).
        Der Einfachheit halber nehmen wir an, dass der vorherige Hidden State
        und der Cell State beide 0 sind (Initialzustand).
        Als Aktivierungsfunktionen nutzen wir die Standardwerte:
        Sigmoid (σ) für die Gates und Tangens Hyperbolicus (tanh) für den Zellzustand.
            1. Die Gate-Berechnungen
                Jedes Gate berechnet sich nach dem Schema:
                    Forget Gate (f[t]):
                        σ(0.5*1 + 0.5*0.5 + 0)= σ(0.75) ~ 0.679
                    Input Gate (i[t]):
                        σ(0.1*1 + 0.1*0.5 + 0)= σ(0.15) ~ 0.537
                    Output Gate (o[t]):
                        σ(0.2*1 + 0.2*0.5 + 0)= σ(0.3) ~ 0.574
            2. Der neue Zellzustand-Kandidat (~C[t])
                Hier nutzen wir tanh für die Skalierung der neuen Informationen:
                    ~C[t]= tanh(0.1*1 + 0.1*0.5 + 0)= tanh(0.15) ~ 0.149
            3. Update des Cell State (C[t])
                Der neue Speicherwert kombiniert das Vergessen des alten Zustands und das Hinzufügen des neuen Inputs:
                    C[t]= (f[t]*C[t-1]) + (i[t]*~C[t])
                    C[t]= (0.679*0) + (0.537*0.149) ~ 0.080
            4. Output / Hidden State (h[t])
                Zum Schluss wird bestimmt, was aus der Zelle nach außen gegeben wird:
                    h[t]= o[t]* tanh(C[t])
                    h[t]= 0.574* tanh(0.080)~ 0.046
            Ergebnis: Der neue Hidden State nach diesem Schritt ist 0.046.

         Führe jetzt die Backpropagation mit Zahlenbeispiel durch
            Das Ziel der Backpropagation ist es, den Fehler (Loss) am Ausgang zu minimieren,
            indem wir die Gewichte basierend auf ihrem Beitrag zum Fehler anpassen.
            Um die Backpropagation für dein LSTM-Beispiel durchzuführen, nehmen wir einen Zielwert (Target) an.
            Wir nutzen den Mean Squared Error (MSE) als Loss-Funktion: E= 0.5* (h[t]-y)^2
.           1. Der Fehler am Output (del_h[t])
                Zuerst berechnen wir, wie stark der Fehler vom aktuellen Hidden State(h[t]) abhängt:
                    del_E/del_h[t]= (h[t]-y)= (0.046-0.5)= -0.454
            2. Gradienten der Gates
                Wir nutzen die Kettenregel, um den Fehler zurück zu den einzelnen Gates zu leiten.
                Wichtig: Die Ableitung der Sigmoid-Funktion σ(x) ist σ(x)*(1-σ(x)).
                Output Gate (o[t]):
                    del_o[t]= del_E/del_h[t]* tanh(C[t])* σ(o[in])* (1-σ(o[in]))
                    del_o[t]= -0.454* tanh(0.080)* 0.574*(1-0.574)
                            ~ -0.454* 0.080* 0.245
                            ~ -0.0089
                Cell State (C[t]):
                    del_C[t]= del_E/del_h[t]* o[t]* (1-tanh²(C[t]))
                    del_C[t]= -0.454* 0.574* (1-0.080²)
                            ~ -0.454* 0.574* 0.993
                            ~ -0.259
                Input Gate (i[t]):
                    del_i[t]= del_C[t]* C[t]* σ(i[in])* (1-σ(i[in]))
                    del_i[t]= -0.259* 0.149* 0.537* (1-0.537)
                            = -0.00959
                Forget Gate (f[t]): (Hier C[t-1]= 0)
         */
    }

    @Test   //darf nur in Kombination mit LSTM_cell().backpropagateBeta() ausgeführt werden, weil ansonsten keine Werte übergeben werden
    public static void toTestB( float[] delO, float[][] wO, float[] delC, float[][] wC,
                                float[] delI, float[][] wI, float[] delFNet,
                                float[][] wF, float[] delLdelC, float[] delUF, float[] delUI, float[] delUC, float[] delUO) {
        float DELTA= 1e-3f;
        //Output Gate (o[t]):
        assertEquals(-0.0089, delO[0], DELTA);                                                                  //Pre-Activation Gradient[Delta_o]
        assertArrayEquals(new float[]{0.2f, 0.2f}, wO[0], DELTA);
        assertEquals(-0.259f, delLdelC[0], DELTA);                                                              //Pre-Activation Gradient[C_t]
        assertEquals(-0.136f, delC[0], DELTA);                                                                  //Pre-Activation Gradient[Delta_C]
        assertArrayEquals(new float[]{0.1136f, 0.1068f}, wC[0], DELTA);
        assertEquals(-0.0096f, delI[0], DELTA);                                                                 //Pre-Activation Gradient[Delta_I]
        assertArrayEquals(new float[]{0.10096f, 0.10048f}, wI[0], DELTA);
        assertEquals(0.0f, delFNet[0], DELTA);                                                                  //0 da der vorherige Cell State 0 war
        assertArrayEquals(new float[]{0.5f, 0.5f}, wF[0], DELTA);
        assertArrayEquals(new float[]{0.10096f, 0.10048f}, wI[0], DELTA);
        assertArrayEquals(new float[]{0.1136f, 0.1068f}, wC[0], DELTA);
        assertArrayEquals(new float[]{0.20089f, 0.20045f}, wO[0], DELTA);
    }

    @Test   //darf nur in Kombination mit LSTM_cell().backpropagateBeta() ausgeführt werden, weil ansonsten keine Werte übergeben werden
    public static void toTestA( float[] delLdelO, float[] delO, float[][] wO, float[] delLdelC, float[] delC, float[][] wC,
                               float[] delI, float[][] wI, float[] delLdelF, float[] delFNet,
                               float[][] wF, float[] delUF, float[] delUI, float[] delUC, float[] delUO) {
        float DELTA= 1e-3f;
        // TODO: 17.04.2026 Wieso wird der Wert 3x nach *[] geschrieben???
        assertEquals(0.0798f, delLdelO[0], DELTA);
        assertEquals(0.0195f, delO[0], DELTA);
        assertArrayEquals(new float[]{0.1981f, 0.1990f}, wO[0], DELTA);
        assertEquals(0.5708f, delLdelC[0], DELTA);
        assertEquals(0.3f, delC[0], DELTA);
        assertArrayEquals(new float[]{0.07f, 0.085f}, wC[0], DELTA);
        assertEquals(0.0211f, delI[0], DELTA);
        assertArrayEquals(new float[]{0.0979f, 0.0989f}, wI[0], DELTA);
        assertEquals(0.5708f, delLdelF[0], DELTA);
        assertEquals(0.1244f, delFNet[0], DELTA);
        assertArrayEquals(new float[]{0.4876f, 0.4938f}, wF[0], DELTA);
        assertEquals(0.1244f*0.5f, delUF[0], DELTA);
        assertEquals(0.0211f*0.5f, delUI[0], DELTA);
        assertEquals(0.3f*0.5f, delUC[0], DELTA);
        assertEquals(0.0195f*0.5f, delUO[0], DELTA);
    }

    @Test
    public void testsBackpropagationB() {
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.5f, 0.5f}};
        float[][]wI= new float[][]{{0.1f, 0.1f}};
        float[][]wC= new float[][]{{0.1f, 0.1f}};
        float[][]wO= new float[][]{{0.2f, 0.2f}};
        LSTM_cell nextNode= new LSTM_cell(new float[]{1f,0f}, new float[]{0f,0f},
                new float[]{1f,0f,0f}, new float[]{0,1,0}, wF, wI, wC, wO,
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f}, 0.1f, true, false, false);

        LSTM_cell prevNode= new LSTM_cell(new float[]{1f}, new float[]{0.5f},
                new float[]{1f,0f,0f}, new float[]{0,1,0}, wF, wI, wC, wO,
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f}, 0.1f, true, false, false);

        float u= 0.1f;
        LSTM_cell node= new LSTM_cell(
                new float[]{0.f, 0.f, 0.f},      //Um Rechnung überschaubar zu halten, setzen wir die Größe des
                new float[]{0.f, 0.f, 0.f},      //  Hidden-/Cell State (Anzahl der Neuronen in der LSTM-Zelle) auf 1.
                new float[]{1f, 0.5f},   //(Ein 2D-Vektor)
                new float[]{0,1,0},
                wF, wI, wC, wO,
                new float[]{0.0f}, new float[]{0.0f},
                new float[]{0.0f}, new float[]{0.0f},
                u, false, false, false);
        node.nextCell = nextNode;
        node.prevCell = prevNode;
        node.forward();
        /* DONE: 17.04.2026 Ich befürchte ich sollte die Backpropagation nochmal komplett neu aufarbeiten
            Zuerst mit einem nachvollziehbaren Bsp. von Google
            "lstm gib mir ein Rechenbeispiel mit Zahlenbeispiel für eine lstm-row mit wF={0.5, 0.5}, wI={0.1, 0.1}, wC={0.1, 0.1}, wO={0.2, 0.2}, bF={0.0, 0.0}, bI={0.0, 0.0}, bC={0.0, 0.0}, bO={0.0, 0.0} und U=0.1 und input={1,0.5}. Verwende bei den Gates auch die Aktivierungsfunktionen"
         */
        node.backpropagateBeta(node.xt, wF, wI, wC, wO,  new float[]{0f}, 1f, 1);

        assertArrayEquals(new float[]{0.4876f, 0.4938f}, wF[0], DELTA);
        assertArrayEquals(new float[]{0.0979f, 0.0989f}, wI[0], DELTA);
        assertArrayEquals(new float[]{0.07f, 0.085f}, wC[0], DELTA);
        assertArrayEquals(new float[]{0.1981f, 0.1990f}, wO[0], DELTA);
    }

    @Test
    public void testsBackpropagationA() {
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.5f, 0.5f}};
        float[][]wI= new float[][]{{0.1f, 0.1f}};
        float[][]wC= new float[][]{{0.1f, 0.1f}};
        float[][]wO= new float[][]{{0.2f, 0.2f}};
        LSTM_cell nextNode= new LSTM_cell(new float[]{1f,0f}, new float[]{0f,0f},
                new float[]{1f,0f,0f}, new float[]{0,1,0}, wF, wI, wC, wO,
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f}, 0.1f, true, false, false);

        LSTM_cell prevNode= new LSTM_cell(new float[]{1f}, new float[]{0f},
                new float[]{1f,0f,0f}, new float[]{0,1,0}, wF, wI, wC, wO,
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f}, 0.1f, true, false, false);

        float u= 0.1f;
        LSTM_cell node= new LSTM_cell(
                new float[]{0.f, 0.f, 0.f},      //Um Rechnung überschaubar zu halten, setzen wir die Größe des
                new float[]{0.f, 0.f, 0.f},      //  Hidden-/Cell State (Anzahl der Neuronen in der LSTM-Zelle) auf 1.
                new float[]{1f, 0.5f},   //(Ein 3D-Vektor)
                new float[]{0,1,0},
                wF, wI, wC, wO,
                new float[]{0.0f}, new float[]{0.0f},
                new float[]{0.0f}, new float[]{0.0f},
                u, false, false, false);
        node.nextCell = nextNode;
        node.prevCell = prevNode;
        node.forward();
        /* DONE: 17.04.2026 Ich befürchte ich sollte die Backpropagation nochmal komplett neu aufarbeiten
            Zuerst mit einem nachvollziehbaren Bsp. von Google
            "lstm gib mir ein Rechenbeispiel mit Zahlenbeispiel für eine lstm-row mit wF={0.5, 0.5}, wI={0.1, 0.1}, wC={0.1, 0.1}, wO={0.2, 0.2}, bF={0.0, 0.0}, bI={0.0, 0.0}, bC={0.0, 0.0}, bO={0.0, 0.0} und U=0.1 und input={1,0.5}. Verwende bei den Gates auch die Aktivierungsfunktionen"
         */
        node.backpropagateBeta(node.xt, wF, wI, wC, wO, new float[]{0f}, 1f, -1);

        assertArrayEquals(new float[]{0.4876f, 0.4938f}, wF[0], DELTA);
        assertArrayEquals(new float[]{0.0979f, 0.0989f}, wI[0], DELTA);
        assertArrayEquals(new float[]{0.07f, 0.085f}, wC[0], DELTA);
        assertArrayEquals(new float[]{0.1981f, 0.1990f}, wO[0], DELTA);
    }

    @Test
    public void testsForwardingA() {
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.5f, 0.5f}};
        float[][]wI= new float[][]{{0.1f, 0.1f}};
        float[][]wC= new float[][]{{0.1f, 0.1f}};
        float[][]wO= new float[][]{{0.2f, 0.2f}};
        LSTM_cell node= new LSTM_cell(
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                new float[]{1f,0.5f}, new float[]{0,1,0}, wF, wI, wC, wO,
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f}, 0.1f, false, false, false);
        node.forward();
        assertEquals(node.forgetGate.getResult()[0], 0.6792f, DELTA);
        assertEquals(node.inputGate.getResult()[0], 0.5374f, DELTA);
        assertEquals(node.candidateGate.getResult()[0], 0.1489f, DELTA);
        assertEquals(node.outputGate.getResult()[0], 0.5744f, DELTA);
        assertArrayEquals(new float[]{0.08f, 0.08f}, node.cellState, DELTA);
        assertArrayEquals(new float[]{0.0459f, 0.0459f}, node.hiddenState, DELTA);
        /* TODO: 15.04.2026 Die Ergebnisse stimmen mit denen des Google-KI-Tests überein
            Das sollte weiter getestet werden, mit anderen Google-KI-Tests
         */
    }
    @Test
    public void testsForwardingB() {
        /* DONE: 16.04.2026 Recherche
            Das interessante bei diesem Google-Testfall ist, dass mir aufgefallen ist
            wie sich in den Errechnungen der Gate-Werte auf den vorherigen HiddenState(h[t-1]) bezogen wird.
            Es gibt für jedes Gate{F,I,C,O} ein <U*>
            <U*> sind rekurrente Gewichte und müssen der Dimensionierung des HiddenStates entspr.
            U[F](Forget Gate): Entscheidet basierend auf dem alten Hidden State, welche Informationen aus dem
                               Langzeitgedächtnis(C[t-1]) gelöscht werden sollen.
            U[I](Input Gate): Bestimmt, wie der alte Hidden State beeinflusst,
                              welche neuen Informationen gespeichert werden.
            U[C](Candidate Gate): Hilft dabei, den neuen Wissens-Kandidaten aus der Kombination von Input und
                                  altem Hidden State zu formen.
            U[O](Output Gate): Steuert, wie der alte Hidden State mitentscheidet,
                               was vom aktuellen Zellzustand nach außen gegeben wird.
            *
            Google fragen: "lstm gib mir ein Rechenbeispiel mit Zahlenbeispiel für eine lstm-row mit wF={0.1, 0.2}, wI={0.9, 0.8}, wC={0.5, 0.5}, wO={0.1, 0.1}, bF=0.5, bI=0, bC=0, bO=0.1 und U=0.1 und input={0.5,0.3}. Verwende bei den Gates auch die Aktivierungsfunktionen"
         */
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.1f, 0.2f}};
        float[][]wI= new float[][]{{0.9f, 0.8f}};
        float[][]wC= new float[][]{{0.5f, 0.5f}};
        float[][]wO= new float[][]{{0.1f, 0.1f}};
        float u= 0.1f;
        LSTM_cell node= new LSTM_cell(
                new float[]{0.0f},      //Um Rechnung überschaubar zu halten, setzen wir die Größe des
                new float[]{0.0f},      //  Hidden-/Cell State (Anzahl der Neuronen in der LSTM-Zelle) auf 1.
                new float[]{0.5f,0.3f},   //(Ein 2D-Vektor)
                new float[]{0,1,0},
                wF, wI, wC, wO,
                new float[]{0.5f}, new float[]{0.f},
                new float[]{0.f}, new float[]{0.1f},
                u, false, false, true);
        node.forward();
        /*DONE Fehler wurde mit [FLAG]<mulMitGetrimmtemInput> in Gate().calculateResult(float,float[],boolean) korrigiert
            IST: node.forgetGate.getResult()[0] kommt raus [0.6456563]
            SOLL: node.forgetGate.getResult()[0] muss ca [0.648] rauskommen
            Bsp-Rechnung:       0.1*0.0 + 0.1*0.5 + 0.2*0.3 + 0.5= 0.61
            Funktion-Rechnung:  0.1*0.0 + 0.2*0.5 +
                                0.1*0.0 +
                                0.5
                                =0.6
         */
        assertEquals(node.forgetGate.getResult()[0], 0.6479f, DELTA);
        assertEquals(node.inputGate.getResult()[0], 0.6659f, DELTA);
        assertEquals(node.candidateGate.getResult()[0], 0.3799f, DELTA);
        assertEquals(node.outputGate.getResult()[0], 0.5449f, DELTA);
        assertArrayEquals(new float[]{0.253f}, node.cellState, DELTA);
        assertArrayEquals(new float[]{0.135f}, node.hiddenState, DELTA);
    }

    @Test
    public void testsForwardingC() {
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.1f, 0.2f}};
        float[][]wI= new float[][]{{0.9f, 0.8f}};
        float[][]wC= new float[][]{{0.5f, 0.5f}};
        float[][]wO= new float[][]{{0.1f, 0.1f}};
        float u= 0.1f;
        LSTM_cell node= new LSTM_cell(
                new float[]{0.0f},      //Um Rechnung überschaubar zu halten, setzen wir die Größe des
                new float[]{0.0f},      //  Hidden-/Cell State (Anzahl der Neuronen in der LSTM-Zelle) auf 1.
                new float[]{0.5f,0.3f},   //(Ein 2D-Vektor)
                new float[]{0,1,0},
                wF, wI, wC, wO,
                new float[]{0.5f}, new float[]{0.f},
                new float[]{0.f}, new float[]{0.1f},
                u, false, false, false);
        node.forward();
        assertEquals(node.forgetGate.getResult()[0], 0.648f, DELTA);
        assertEquals(node.inputGate.getResult()[0], 0.666f, DELTA);
        assertEquals(node.candidateGate.getResult()[0], 0.380f, DELTA);
        assertEquals(node.outputGate.getResult()[0], 0.545f, DELTA);
        assertArrayEquals(new float[]{0.253f}, node.cellState, DELTA);
        assertArrayEquals(new float[]{0.135f}, node.hiddenState, DELTA);
    }

    @Test
    public void testsForwardingD() {
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.2f, 0.3f, 0.1f}};
        float[][]wI= new float[][]{{0.1f, -0.1f, 0.2f}};
        float[][]wC= new float[][]{{-0.3f, 0.1f, 0.5f}};
        float[][]wO= new float[][]{{0.4f, 0.2f, -0.1f}};
        float u= 0.1f;
        LSTM_cell node= new LSTM_cell(
                new float[]{0.f, 0.f, 0.f},      //Um Rechnung überschaubar zu halten, setzen wir die Größe des
                new float[]{0.f, 0.f, 0.f},      //  Hidden-/Cell State (Anzahl der Neuronen in der LSTM-Zelle) auf 1.
                new float[]{0.5f,-0.2f, 0.1f},   //(Ein 3D-Vektor)
                new float[]{0,1,0},
                wF, wI, wC, wO,
                new float[]{0.1f}, new float[]{-0.1f},
                new float[]{0.f}, new float[]{0.2f},
                u, false, false, false);
        node.forward();
        assertEquals(node.forgetGate.getResult()[0], 0.537f, DELTA);
        assertEquals(node.inputGate.getResult()[0], 0.497f, DELTA);
        assertEquals(node.candidateGate.getResult()[0], -0.119f, DELTA);
        assertEquals(node.outputGate.getResult()[0], 0.587f, DELTA);
        assertEquals(node.cellState[0], -0.059f, DELTA);       /* DONE: 16.04.2026 wieso enthält node.cellState[] mehr als einen Wert???
                                                                      *       evt weil preCS 3 dimensional ist*/
        assertEquals(node.hiddenState[0], -0.0346f, DELTA);    /* DONE: 16.04.2026 wieso enthält node.hiddenState[] mehr als einen Wert???
                                                                      *       evt weil preHS 3 dimensional ist*/
    }

    @Test
    public void testsForwardingE() {
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.1f, -0.2f, 0.5f}};
        float[][]wI= new float[][]{{0.2f, 0.3f, -0.1f}};
        float[][]wC= new float[][]{{0.8f, -0.5f, 0.0f}};
        float[][]wO= new float[][]{{0.4f, 0.1f, 0.2f}};
        float u= 0.1f;
        LSTM_cell node= new LSTM_cell(
                new float[]{0.2f},      //Um Rechnung überschaubar zu halten, setzen wir die Größe des
                new float[]{0.1f},      //  Hidden-/Cell State (Anzahl der Neuronen in der LSTM-Zelle) auf 1.
                new float[]{0.5f,-0.2f},   //(Ein 2D-Vektor)
                new float[]{0,1,0},
                wF, wI, wC, wO,
                new float[]{0.1f}, new float[]{0.2f},
                new float[]{0.f}, new float[]{-0.1f},
                u, true, false, true);
        node.forward();
        assertEquals(node.forgetGate.getResult()[0], 0.56f, DELTA);
        assertEquals(node.inputGate.getResult()[0], 0.557f, DELTA);
        assertEquals(node.candidateGate.getResult()[0], 0.462f, DELTA);
        assertEquals(node.outputGate.getResult()[0], 0.525f, DELTA);
        assertEquals(node.cellState[0], 0.369f, DELTA);       /* DONE: 16.04.2026 wieso enthält node.cellState[] mehr als einen Wert???
         *       evt weil preCS 3 dimensional ist*/
//        assertArrayEquals(new float[]{-0.0181f}, node.cellState, DELTA);
        assertEquals(node.hiddenState[0], 0.185f, DELTA);    /* DONE: 16.04.2026 wieso enthält node.hiddenState[] mehr als einen Wert???
         *       evt weil preHS 3 dimensional ist*/
//        assertArrayEquals(new float[]{-0.0181f}, node.hiddenState, DELTA);
    }
}
