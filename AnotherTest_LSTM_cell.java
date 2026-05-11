package lstm2605111326;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class AnotherTest_LSTM_cell {
    Func TANH= x-> (float)((Math.exp(x[0])-Math.exp(-x[0])) / (Math.exp(x[0])+Math.exp(-x[0])));

    @Test
    public void testsMultipleTimestepsForwardingAndBackpropagationE() {
        /* TODO: 07.05.2026
            An Google(gemini)
                "LSTM: Ein Rechenbeispiel sowohl für Forwarding als auch für Backpropagation, mit Inputs x={1.0, 0.5, 2.0}, rekurrente Gewicht U={Uf=0.1, Ui=0.2, Uc=0.3, Uo=0.4}, Gewicht W={Wf=0.5, Wi=0.6, Wc=0.7, Wo=0.8}, cs[t-1]= 0.0, hs[t-1]=0.0 und Bias b=0.1 wobei ich die Verwendung der rekurrenten Gewichte(Uf,Ui,Uc,Uo) nachvollziehen kann. Ein Rechenbeispiel mit 3 Zeitschritte."
                "Mach ein konkretes Zahlenbeispiel für den Backpropagation-Schritt mit einem target von 0.5"
                "Ich erhalte für h3=0.7083, was habe ich falsch gemacht?"
                "Ich erhalte für c3=1.1618077, was habe ich falsch gemacht?"
                "bitte rechne das nochmal nach mit h3=0.7083 und c3=1.1618077"
                "Berechne mir die Gradienten aller Gates für die Backpropagation von TimeStep 3"
                "Berechne mir die Gradienten aller Gates für die Backpropagation von TimeStep 2"
                "Berechne mir die Gradienten aller Gates für die Backpropagation von TimeStep 1"
                "aktualisiere/optimiert die Gewichte mit ETA=0.01"
            file:///C:/Users/sven-.SVEN/Desktop/GoogleLstmTestBspBackForwardRechenbeispielMitRekurrentenGewichtenDreiTimesteps/LSTM_RechenbspForwardingBackpropagationMitRekurrentenGradientenDreiTimestepsVollstaendig/LSTM_R~1.HTM
        */
        float DELTA = 1e-3f;
        Func SIGMOID = x -> (float) (1 / (1 + Math.exp(-x[0])));                                                // TODO: 11.05.2026 ist das so richtig?
        Func SIGMOID_DERIVATIVE = x -> (x[0] * (1 - x[0]));                                                     // TODO: 11.05.2026 ist das so richtig?
        Func TANH = x -> (float) ((Math.exp(x[0]) - Math.exp(-x[0])) / (Math.exp(x[0]) + Math.exp(-x[0])));     // TODO: 11.05.2026 ist das so richtig?
        Func TANH_DERIVATIVE = x -> (1.0 - (Math.pow(x[0], 2)));                                                // TODO: 11.05.2026 ist das so richtig?
        float ETA = 0.01f;

        float[] inputs = new float[]{1.0f, 0.5f, 2.0f};   //x0, x1, x2
        ArrayList<float[]> cs = new ArrayList<>();
        float[] hs_pre = new float[]{0.0f};
        float[] cs_pre = new float[]{0.0f};
        cs.add(cs_pre);
        ArrayList<float[]> hs = new ArrayList<>();
        hs.add(hs_pre);

        float w_f = 0.5f;
        float w_i = 0.6f;
        float w_c = 0.7f;
        float w_o = 0.8f;
        float u_f = 0.1f;
        float u_i = 0.2f;
        float u_c = 0.3f;
        float u_o = 0.4f;
        float BIAS = 0.1f;

        float[] Xt = new float[hs.get(hs.size() - 1).length + inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            Xt[i] = inputs[i];                           //nach Xt[] werden zuerst die input-Werte aus xt[] geschrieben...
        }

        //Forward Propagation (Timestep 1)
        int t = 0;       //TimeStep
        Gate forgetGate = new Gate(SIGMOID, Xt, new float[][]{{w_f}}, new float[]{BIAS}, u_f, hs.get(hs.size() - 1), t);
        Gate inputGate = new Gate(SIGMOID, Xt, new float[][]{{w_i}}, new float[]{BIAS}, u_i, hs.get(hs.size() - 1), t);
        Gate candidateGate = new Gate(TANH, Xt, new float[][]{{w_c}}, new float[]{BIAS}, u_c, hs.get(hs.size() - 1), t);
        Gate outputGate = new Gate(SIGMOID, Xt, new float[][]{{w_o}}, new float[]{BIAS}, u_o, hs.get(hs.size() - 1), t);

        cs.set(t, calcNewCS(cs.get(hs.size() - 1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.set(t, calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.646, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.668, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.664, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.711, outputGate.getResult(t)[0], DELTA);
        assertEquals(0.444, cs.get(t)[0], DELTA);
        assertEquals(0.296, hs.get(t)[0], DELTA);

        //Forward Propagation (Timestep 2)
        t = 1;       //TimeStep
        forgetGate.calculateResult(u_f, hs.get(t - 1), t);
        inputGate.calculateResult(u_i, hs.get(t - 1), t);
        candidateGate.calculateResult(u_c, hs.get(t - 1), t);
        outputGate.calculateResult(u_o, hs.get(t - 1), t);

        cs.add(calcNewCS(cs.get(t - 1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.add(calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.594, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.6128, inputGate.getResult(t)[0], 1e-4f);
        assertEquals(0.4921, candidateGate.getResult(t)[0], 1e-4f);
        assertEquals(0.6499, outputGate.getResult(t)[0], 1e-4f);
        assertEquals(0.5651, cs.get(t)[0], 1e-4f);
        assertEquals(0.3326, hs.get(t)[0], 1e-4f);

        //Forward Propagation (Timestep 3)
        t = 2;       //TimeStep
        forgetGate.calculateResult(u_f, hs.get(t-1), t);
        inputGate.calculateResult(u_i, hs.get(t-1), t);
        candidateGate.calculateResult(u_c, hs.get(t-1), t);
        outputGate.calculateResult(u_o, hs.get(t-1), t);

        cs.add(calcNewCS(cs.get(t - 1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.add(calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.756, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.797, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.922, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.862, outputGate.getResult(t)[0], DELTA);
        assertEquals(1.163, cs.get(t)[0], 1e-2f);
        assertEquals(0.709, hs.get(t)[0], DELTA);

        //Backpropagation through Time (BPTT) Ausgangslage (Timestep 3)
        t = 2;       //TimeStep
        //1. Der Fehler am Ausgang <delH3>
        float y= 0.5f;      //Target
        float delLdelH3= hs.get(t)[0]- y;
        float delH3= delLdelH3;
        assertEquals(0.2083, delLdelH3, DELTA);
        //2. Gradient des Output Gates <delO3>
        float delLdelO3= (float)(delLdelH3* TANH.calc(cs.get(t)[0]));
        assertEquals(0.17116, delLdelO3, DELTA);
        float delO3= (float)(delH3* TANH.calc(cs.get(t)[0])* SIGMOID_DERIVATIVE.calc(outputGate.getResult(t)[0]));
        assertEquals(0.0204, delO3, DELTA);
        //3. Gradient des Cell States <delC3>
        float delLdelC3= (float)(delH3* outputGate.getResult(t)[0]* (1-Math.pow(TANH.calc(cs.get(t)[0]), 2)));
        assertEquals(0.058, delLdelC3, DELTA);
        //4. Gradienten für das Eingangs-Gewicht <delLdelWo>
        float delLdelWo= delO3* inputs[t];
        assertEquals(0.04072, delLdelWo, DELTA);
        //4. Gradienten für das rekurrente Gewicht <delLdelUo>
        float delLdelUo= delO3* hs.get(t-1)[0];
        assertEquals(0.0066, delLdelUo, DELTA);

        //1. Bekannte Werte(Timestep 3)
        assertEquals(0.2083, delH3, DELTA);
        assertEquals(0.0583, delLdelC3, DELTA);
        assertEquals(0.862, outputGate.getResult(t)[0], DELTA);
        assertEquals(0.566, cs.get(t-1)[0], DELTA);
        assertEquals(0.797, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.922, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.756, forgetGate.getResult(t)[0], DELTA);

        //2. Gradienten der fehlenden Gates (vor der Aktivierung)
        //Gradient des Forget Gates <delF3>
        float delF3= (float)(delLdelC3* cs.get(t-1)[0]* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(t)[0]));
        assertEquals(0.0061, delF3, 1e-4f);
        //Gradient des Input Gates <delI3>
        float delI3= (float)(delLdelC3* candidateGate.getResult(t)[0]* SIGMOID_DERIVATIVE.calc(inputGate.getResult(t)[0]));
        assertEquals(0.0087, delI3, 1e-4f);
        //Gradient des Candidate Gates <delCan3>
        float delCan3= (float)(delLdelC3* inputGate.getResult(t)[0]* TANH_DERIVATIVE.calc(candidateGate.getResult(t)[0]));
        assertEquals(0.007, delCan3, 1e-4f);

        //3. Zusammenfassung der Gate-Gradienten
        //Diese δ-Werte geben an, in welche Richtung die Gates korrigiert werden müssen
        assertEquals(0.0204, delO3, 1e-4f);     //Das Output-Gate war etwas zu weit offen
        assertEquals(0.0061, delF3, 1e-4f);     //Das Forget-Gate hat etwas zu viel Information aus der Vergangenheit durchgelassen.
        assertEquals(0.0087, delI3, 1e-4f);     //Das Input-Gate hat etwas zu viel neuen Input hineingelassen.

        float delUo= delO3* hs.get(t-1)[0];
        assertEquals(0.00678, delUo, 1e-4f);
        float delUf= delF3* hs.get(t-1)[0];
        assertEquals(0.00203, delUf, 1e-4f);
        float delUi= delI3* hs.get(t-1)[0];
        assertEquals(0.0029, delUi, 1e-4f);
        float delUcan= delCan3* hs.get(t-1)[0];
        assertEquals(0.00232, delUcan, 1e-4f);

        //1. Vorbereitung: Der kumulierte Fehler für <h2> und <c2>
        //Der Fehler gelangt über zwei Wege von Schritt 3 zurück zu Schritt 2:
        //Der Fehler von der Zelle <delC2_fromC3>:
        float delLdelC2= delLdelC3* forgetGate.getResult(t)[0];
        assertEquals(0.04411, delLdelC2, 1e-4f);
        //Der Fehler vom Hidden State <delH2_fromH3>
        float delLdelH2= (delF3*u_f) +
                (delI3*u_i) +
                (delCan3*u_c) +
                (delO3*u_o);
        assertEquals(0.01261, delLdelH2, 1e-4f);
        //2. Gradienten der Gates für t=2   (TimeStep: 3)
        //Werte für t=2 (Forward Pass):
        t=1;
        assertEquals(0.594, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.613, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.492, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.65, outputGate.getResult(t)[0], DELTA);
        assertEquals(0.566, cs.get(t)[0], DELTA);
        assertEquals(0.444, cs.get(t-1)[0], DELTA);
        //Zell-Fehler <delC2> (Gesamt):
        //Zusätzlich zum Fehler aus der Zelle <delC2_fromC3> kommt nun der Anteil vom Hidden State <hs.get(2)[0]> hinzu:
        float delH2= delLdelH2;
        float delC2= (float)(delLdelC2+ (delLdelH2* outputGate.getResult(t)[0]* TANH_DERIVATIVE.calc(TANH.calc(cs.get(t)[0]))));
        assertEquals(0.05014, delC2, 1e-4f);
        //Berechnung der einzelnen Gates
        //Gradient des Output Gates <delO2>
        float delO2= (float)(delH2* TANH.calc(cs.get(t)[0])* SIGMOID_DERIVATIVE.calc(outputGate.getResult(t)[0]));
        assertEquals(0.00147, delO2, 1e-4f);
        //Gradient des Forget Gates <delF2>
        float delF2= (float)(delC2* cs.get(t-1)[0]* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(t)[0]));
        assertEquals(0.00537, delF2, 1e-4f);
        //Gradient des Input Gates <delI2>
        float delI2= (float)(delC2* candidateGate.getResult(t)[0]* SIGMOID_DERIVATIVE.calc(inputGate.getResult(t)[0]));
        assertEquals(0.00585, delI2, 1e-4f);
        //Gradient des Candidate Gates <delCan2>
        float delCan2= (float)(delC2* inputGate.getResult(t)[0]* TANH_DERIVATIVE.calc(candidateGate.getResult(t)[0]));
        assertEquals(0.02330, delCan2, 1e-4f);
        //3. Zusammenfassung der Gewichtsgradienten für t=2
        assertEquals(0.5, inputs[t], 1e-4f);
        assertEquals(0.296, hs.get(t-1)[0], DELTA);

        //Für die Input Gewichte
        //Output Gates <delWO2>
        float delWO2= delO2*inputs[t];
        assertEquals(0.00073, delWO2, 1e-5f);
        //Output Gates <delWF2>
        float delWF2= delF2*inputs[t];
        assertEquals(0.00269, delWF2, 1e-5f);
        //Output Gates <delWI2>
        float delWI2= delI2*inputs[t];
        assertEquals(0.00293, delWI2, 1e-5f);
        //Output Gates <delWCan2>
        float delWCan2= delCan2*inputs[t];
        assertEquals(0.01164, delWCan2, 1e-4f);

        //Für die rekurrenten Gewichte
        //Output Gates <delWO2>
        float delUO2= delO2*hs.get(t-1)[0];
        assertEquals(0.00043, delUO2, 1e-5f);
        //Output Gates <delWF2>
        float delUF2= delF2*hs.get(t-1)[0];
        assertEquals(0.00159, delUF2, 1e-5f);
        //Output Gates <delWI2>
        float delUI2= delI2*hs.get(t-1)[0];
        assertEquals(0.00173, delUI2, 1e-5f);
        //Output Gates <delWCan2>
        float delUCan2= delCan2*hs.get(t-1)[0];
        assertEquals(0.00689, delUCan2, 1e-4f);

        //Backpropagation through Time (BPTT) Ausgangslage (t=1)
        t= 0;       //TimeStep
        //1. Kumulierter Fehler für h1 und c1
        //Fehler vom Hidden State h2 to h1):
        //Wir nutzen die Gate-Gradienten von t=2 (<delF2>, <delI2>, <delCan2>, <delO2>) und
        // die rekurrenten Gewichte (<u_f>, <u_i>, <u_c>, <u_o>):
        float delLdelH1= (delF2*u_f) + (delI2*u_i) + (delCan2*u_c) + (delO2*u_o);
        assertEquals(0.009285, delLdelH1, DELTA);
        //Fehler vom Cell State c2 to c1):
        //Der Fehler wird durch das Forget Gate von Schritt 2 skaliert:
        float delLdelC1= delC2* forgetGate.getResult(1)[0];
        assertEquals(0.02978, delLdelC1, DELTA);
        //2. Gradienten der Gates für t=1
        //Werte für t=1 (Forward Pass):
        assertEquals(0.711, outputGate.getResult(t)[0], DELTA);
        assertEquals(0.444, cs.get(t)[0], DELTA);
        assertEquals(0.0, cs_pre[0], DELTA);
        assertEquals(0.668, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.664, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.646, forgetGate.getResult(t)[0], DELTA);
        //Gesamt-Zellfehler delC1:
        float delC1_total= (float)(delLdelC1+
                (delLdelH1* outputGate.getResult(t)[0]* TANH_DERIVATIVE.calc(TANH.calc(cs.get(t)[0])))
        );
        assertEquals(0.03523, delC1_total, 1e-4f);

        //Berechnung der Gate-Gradienten delGate_t=1:
        //Output-Gate
        float delO1= (float)(delLdelH1* TANH.calc(cs.get(t)[0])* SIGMOID_DERIVATIVE.calc(outputGate.getResult(t)[0]));
        assertEquals(0.000795, delO1, 1e-5f);
        //Forget-Gat
        float delF1=(float)(delC1_total* cs_pre[0] * SIGMOID_DERIVATIVE.calc(forgetGate.getResult(t)[0]));
        assertEquals(0.0, delF1, 1e-5f);
        //Input-Gate
        float delI1=(float)(delC1_total* candidateGate.getResult(t)[0]* SIGMOID_DERIVATIVE.calc(inputGate.getResult(t)[0]));
        assertEquals(0.00519, delI1, 1e-4f);
        //Candidate-Gate
        float delCan1=(float)(delC1_total* inputGate.getResult(t)[0]* (1-Math.pow(candidateGate.getResult(t)[0], 2)));
        assertEquals(0.01316, delCan1, 1e-4f);

        //Gewichtsgradienten für t=1 (x1=1.0, h0=0.0)
        assertEquals(1.0, inputs[t], 1e-4f);
        assertEquals(0.0, hs_pre[0], DELTA);
        //deltaWeight
        //Output Gate
        float delWO1= delO1*inputs[t];
        assertEquals(0.00079, delWO1, 1e-5f);
        //Output Gat
        float delWF1= delF1*inputs[t];
        assertEquals(0.0, delWF1, 1e-5f);
        //Output Gate
        float delWI1= delI1*inputs[t];
        assertEquals(0.00518, delWI1, 1e-4f);
        //Output Gate
        float delWCan1= delCan1*inputs[t];
        assertEquals(0.01315, delWCan1, 1e-4f);

        /* DONE: 10.05.2026
            Das muss überprüft werden, denn der jeweilige rekurrente Gradient bildet sich
            aus der Summe der Gradienten über alle Zeitschritte.
            Zu diesem Zeitpunkt SOLL hs[] so aussehen   {0.333, 0.296, 0}
                                IST                     {0.70834327, 0.33255807, 0.29625815}
            BEOBACHTUNG:    REDUNDANT an IST[0]
                            SOLL[0] steht an IST[1]
                            SOLL[1] steht an IST[2]
                            SOLL[2] nicht vorhanden
            ERKENNTNIS:     Meine Implementierung erwartet, dass der 1. HiddenState von hs_pre bezogen wird
        */
        // UPDATE der rekurrenten Gewichte: 	Neues Gewicht (U_alt −ETA* Gradient)
        float uF_total= (delF3* hs.get(1)[0])+
                (delF2* hs.get(0)[0])+
                (delF1* hs_pre[0]);
        assertEquals(0.00362, uF_total, 1e-4f);
        u_f-= ETA* uF_total;
        assertEquals(0.09996, u_f, 1e-5f);

        float uI_total= (delI3* hs.get(1)[0])+
                (delI2* hs.get(0)[0])+
                (delI1* hs_pre[0]);
        assertEquals(0.00463, uI_total, 1e-4f);
        u_i-= ETA* uI_total;
        assertEquals(0.19995, u_i, 1e-5f);

        float uCan_total=   (delCan3* hs.get(1)[0])+
                (delCan2* hs.get(0)[0])+
                (delCan1* hs_pre[0]);
        assertEquals(0.00923, uCan_total, 1e-4f);    //Meine Implementierung erwartet, dass der 1. HiddenState von hs_pre bezogen wird
        u_c-= ETA* uCan_total;
        assertEquals(0.29991, u_c, 1e-5f);

        float uO_total= (delO3* hs.get(1)[0])+
                (delO2* hs.get(0)[0])+
                (delO1* hs_pre[0]);
        assertEquals(0.00723, uO_total, 1e-4f);
        u_o-= ETA* uO_total;
        assertEquals(0.39993, u_o, 1e-5f);

        //Update der Input-Gewichte (W)
        //Forget Gate
        float wF_total= delF3*inputs[2]+ delF2*inputs[1]+ delF1*inputs[0];
        assertEquals(0.01489, wF_total, 1e-4f);
        w_f-= ETA* wF_total;
        assertEquals(0.49985, w_f, 1e-5f);
        //Input Gate
        float wI_total= delI3*inputs[2]+ delI2*inputs[1]+ delI1*inputs[0];
        assertEquals(0.02552, wI_total, 1e-4f);
        w_i-= ETA* wI_total;
        assertEquals(0.59974, w_i, 1e-5f);
        //Candidate Gate
        float wCan_total= delCan3*inputs[2]+ delCan2*inputs[1]+ delCan1*inputs[0];
        assertEquals(0.03881, wCan_total, 1e-4f);
        w_c-= ETA* wCan_total;
        assertEquals(0.69961, w_c, 1e-5f);
        //Output Gate
        float wO_total= delO3*inputs[2]+ delO2*inputs[1]+ delO1*inputs[0];
        assertEquals(0.04233, wO_total, DELTA);
        w_o-= ETA* wO_total;
        assertEquals(0.79958, w_o, 1e-5f);

        //Update des Bias (b)
        //Für den Bias (angenommen b ist für alle Gates gleich) addieren wir einfach alle Gate-Deltas auf:
        //BIAS= SUM{TimeSteps{Gates{delta}}
        float BIAS_total=   delO3+ delF3+ delI3+ delCan3 +
                delO2+ delF2+ delI2+ delCan2 +
                delO1+ delF1+ delI1+ delCan1;
        assertEquals(0.0937, BIAS_total, 1e-2f);
        BIAS-= ETA* BIAS_total;
        assertEquals(0.09906, BIAS, DELTA);

        /* TODO: 08.05.2026 Wann werden die Gewichte -hinsichtlich der Gradienten und ETA- aktualisiert/optimiert?
            Die Aktualisierung erfolgt typischerweise nachdem eine komplette Sequenz (oder ein Batch von Sequenzen)
            verarbeitet wurde und der Fehler (Loss) berechnet wurde.
            Die Gewichte werden nicht bei jedem einzelnen Zeitschritt der Sequenz geupdatet,
            sondern stapelweise (batch-wise) am Ende der Backpropagation.
            Weil es sich hierbei um SharedWeights handelt.
            C:\Users\sven-.SVEN\Desktop\GoogleLstmTestBspBackForwardRechenbeispielMitRekurrentenGewichtenDreiTimesteps\LSTM_KomplettesRechenbeispiel\LSTM_Komplettes.htm
         */
    }

    @Test
    public void testsMultipleTimestepsForwardingAndBackpropagationD() {
        /* TODO: 07.05.2026
            Mir reichts!
            Ich habe die befürchtung chatGPT erklärt das zu sprunghaft und überfliegt einige Schritte
            Gemini erklärt das verständlicher
            *
            Nochmal mit Gemini versuchen
            An Google(gemini)
                "LSTM: Ein Rechenbeispiel sowohl für Forwarding als auch für Backpropagation, mit Inputs x={1.0, 0.5, 2.0}, rekurrente Gewicht U={Uf=0.1, Ui=0.2, Uc=0.3, Uo=0.4}, Gewicht W={Wf=0.5, Wi=0.6, Wc=0.7, Wo=0.8}, cs[t-1]= 0.0, hs[t-1]=0.0 und Bias b=0.1 wobei ich die Verwendung der rekurrenten Gewichte(Uf,Ui,Uc,Uo) nachvollziehen kann. Ein Rechenbeispiel mit 3 Zeitschritte."
                "Mach ein konkretes Zahlenbeispiel für den Backpropagation-Schritt mit einem target von 0.5"
                "Ich erhalte für h3=0.7083, was habe ich falsch gemacht?"
                "Ich erhalte für c3=1.1618077, was habe ich falsch gemacht?"
                "bitte rechne das nochmal nach mit h3=0.7083 und c3=1.1618077"
        */
        float DELTA= 1e-3f;
        Func SIGMOID= x-> (float)(1 / (1 + Math.exp(-x[0])));
        Func SIGMOID_DERIVATIVE= x->(x[0]*(1-x[0]));
        Func TANH= x-> (float)((Math.exp(x[0])-Math.exp(-x[0])) / (Math.exp(x[0])+Math.exp(-x[0])));
        Func TANH_DERIVATIVE= x->(1.0 - (Math.pow(x[0], 2)));
        float ETA= 0.01f;
        //Google: LSTM: Ein Rechenbeispiel sowohl für Forwarding als auch für Backpropagation, wobei ich die Verwendung der rekurrenten Gewichte(Uf,Ui,Uc,Uo) nachvollziehen kann. Ein Rechenbeispiel mit 3 Zeitschritte.

        float[]inputs= new float[]{1.0f, 0.5f, 2.0f};   //x0, x1, x2
        ArrayList<float[]> cs= new ArrayList<>();
        float[]hs_pre= new float[]{0.0f};
        float[]cs_pre= new float[]{0.0f};
        cs.add(cs_pre);              // DONE: 29.04.2026 es ist notwendig dass ich die HS und CS pro timestep merke => there is no need for <cs_pre> anymore
        ArrayList<float[]> hs= new ArrayList<>();
        hs.add(hs_pre);              // DONE: 29.04.2026 es ist notwendig dass ich die HS und CS pro timestep merke => there is no need for <hs_pre> anymore

        float w_f= 0.5f;
        float w_i= 0.6f;
        float w_c= 0.7f;
        float w_o= 0.8f;
        float u_f= 0.1f;
        float u_i= 0.2f;
        float u_c= 0.3f;
        float u_o= 0.4f;
        float BIAS= 0.1f;

        float[]Xt= new float[hs.get(hs.size()-1).length+ inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            Xt[i] = inputs[i];                           //nach Xt[] werden zuerst die input-Werte aus xt[] geschrieben...
        }

        //Forward Propagation (Timestep 1)
        int t= 0;       //TimeStep
        Gate forgetGate= new Gate(SIGMOID, Xt, new float[][]{{w_f}}, new float[]{BIAS}, u_f, hs.get(hs.size()-1), t);
        Gate inputGate= new Gate(SIGMOID, Xt, new float[][]{{w_i}}, new float[]{BIAS}, u_i, hs.get(hs.size()-1), t);
        Gate candidateGate= new Gate(TANH, Xt, new float[][]{{w_c}}, new float[]{BIAS}, u_c, hs.get(hs.size()-1), t);
        Gate outputGate= new Gate(SIGMOID, Xt, new float[][]{{w_o}}, new float[]{BIAS}, u_o, hs.get(hs.size()-1), t);

        cs.set(t,calcNewCS(cs.get(hs.size()-1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.set(t,calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.646, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.668, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.664, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.711, outputGate.getResult(t)[0], DELTA);
        assertEquals(0.444, cs.get(t)[0], DELTA);
        assertEquals(0.296, hs.get(t)[0], DELTA);

        //Forward Propagation (Timestep 2)
        t= 1;       //TimeStep
        forgetGate.calculateResult(u_f, hs.get(t-1), t);
        inputGate.calculateResult(u_i, hs.get(t-1), t);
        candidateGate.calculateResult(u_c, hs.get(t-1), t);
        outputGate.calculateResult(u_o, hs.get(t-1), t);

        cs.add(calcNewCS(cs.get(t-1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.add(calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.594, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.6128, inputGate.getResult(t)[0], 1e-4f);
        assertEquals(0.4921, candidateGate.getResult(t)[0], 1e-4f);
        assertEquals(0.6499, outputGate.getResult(t)[0], 1e-4f);
        assertEquals(0.5651, cs.get(t)[0], 1e-4f);
        assertEquals(0.3326, hs.get(t)[0], 1e-4f);

        //Forward Propagation (Timestep 3)
        t= 2;       //TimeStep
        forgetGate.calculateResult(u_f, hs.get(t-1), t);
        inputGate.calculateResult(u_i, hs.get(t-1), t);
        candidateGate.calculateResult(u_c, hs.get(t-1), t);
        outputGate.calculateResult(u_o, hs.get(t-1), t);

        cs.add(calcNewCS(cs.get(t-1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.add(calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.7564, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.7968, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.9216, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.8621, outputGate.getResult(t)[0], DELTA);
        assertEquals(1.1618, cs.get(t)[0], DELTA);
        assertEquals(0.7083, hs.get(t)[0], DELTA);

        //Backpropagation through Time (BPTT) Ausgangslage (t=3)
        t= 2;       //TimeStep
        float TARGET= 0.5f;
        float L= (float)((0.5)*Math.pow(hs.get(t)[0]-TARGET, 2));//Loss nach MSE
        assertEquals(0.0216f, L, DELTA);
        float delLdelH3= hs.get(t)[0]-TARGET;                             //Der Fehlergradient am Output (Delta(h3))
        assertEquals(0.2083f, delLdelH3, DELTA);
        //Zuerst berechnen wir den Einfluss auf das Output Gate:
        TANH.calc(cs.get(t)[0]);                                          //Anteil am Cell State
        float delLdelO3= (float)(delLdelH3* TANH.calc(cs.get(t)[0]));     //Gradient für Output-Gate (o3)
        assertEquals(0.1711f, delLdelO3, DELTA);
        float delO3= delLdelO3* outputGate.getResult(t)[0]* (1-outputGate.getResult(t)[0]);        //Gewichtsupdate Wo
        assertEquals(0.02034f, delO3, DELTA);

        //Gradient für den Cell State (c3)
        //Hier fließt der Fehler von h3 durch die tanh-Funktion in den Zellzustand:
        float delLdelC3= (float)(delLdelH3* outputGate.getResult(t)[0]* (1-(Math.pow(TANH.calc(cs.get(t)[0]),2))));
        assertEquals(0.05837f, delLdelC3, DELTA);

        //Forget Gate (f3):
        float delLdelF3= delLdelC3* cs.get(1)[0];
        assertEquals(0.03298f, delLdelF3, DELTA);
        float delF3= (float)(delLdelF3* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(t)[0]));
        assertEquals(0.00608f, delF3, DELTA);
        //Input Gate (i3):
        float delLdelI3= delLdelC3* candidateGate.getResult(t)[0];
        assertEquals(0.05378f, delLdelI3, DELTA);
        float delI3= (float)(delLdelI3* SIGMOID_DERIVATIVE.calc(inputGate.getResult(t)[0]));
        assertEquals(0.00871f, delI3, DELTA);
        //Fehler am Candidate Gate (c[3])
        float delC3= (float)(delLdelC3* inputGate.getResult(2)[0]* TANH_DERIVATIVE.calc(candidateGate.getResult(2)[0]));
        assertEquals(0.007f, delC3, DELTA);
        //Gradient für CellState
        //Fehler am Output
        float delH3= hs.get(t)[0]-TARGET;
        delO3= (float)(delH3* TANH.calc(cs.get(t)[0])* SIGMOID_DERIVATIVE.calc(outputGate.getResult(t)[0]));
        assertEquals(0.02034f, delO3, DELTA);
        float delCs3= (float)(delH3* outputGate.getResult(t)[0]* (1-Math.pow(TANH.calc(cs.get(t)[0]), 2)));
        assertEquals(0.05835f, delCs3, DELTA);

        //Update der Gewichte W und U (für Zeitschritt 3)
        float delLdelWo= delO3* inputs[t];                   //Gradient zum Updaten des Gewichts (Wo)
        assertEquals(0.0406f, delLdelWo, DELTA);
        float delLdelWf= delF3* inputs[t];                   //Gradient zum Updaten des Gewichts (Wf)
        assertEquals(0.0122f, delLdelWf, DELTA);
        float delLdelWi= delI3* inputs[t];                   //Gradient zum Updaten des Gewichts (Wi)
        assertEquals(0.0174f, delLdelWi, DELTA);
        float delLdelWc= delC3* inputs[t];                   //Gradient zum Updaten des Gewichts (Wc)
        assertEquals(0.014f, delLdelWc, DELTA);

        float delLdelUo= delO3* hs.get(1)[0];                   //Gradient zum Updaten des rekurrenten Gewichts (Uo)
        assertEquals(0.00676f, delLdelUo, DELTA);
        float delLdelUf3= delF3* hs.get(1)[0];                   //Gradient zum Updaten des rekurrenten Gewichts (Uf)
        assertEquals(0.00202f, delLdelUf3, DELTA);
        float delLdelUf_total= 0;
        delLdelUf_total+= delLdelUf3;
        float delLdelUi= delI3* hs.get(1)[0];                   //Gradient zum Updaten des rekurrenten Gewichts (Ui)
        assertEquals(0.0029f, delLdelUi, DELTA);
        float delLdelUc= delC3* hs.get(1)[0];                   //Gradient zum Updaten des rekurrenten Gewichts (Uc)
        assertEquals(0.00235f, delLdelUc, DELTA);


        float delLdelC2= delLdelC3* forgetGate.getResult(t)[0];
        assertEquals(0.04414f, delLdelC2, DELTA);
        //Fehlerfluss zum vorigen Hidden State delLdelH2
        /*Das ist der komplexere Teil.
         *Der Hidden State hat im Schritt alle vier Gates beeinflusst (über die rekurrenten Gewichte).
         *Daher summiert sich der Fehler aus allen vier Deltas auf:*/
        float delLdelH2= delF3*u_f + delI3*u_i + delC3*u_c + delO3*u_o;
        assertEquals(0.012589f, delLdelH2, DELTA);
        float delLdelC2_gesamt= (float)(delLdelC2+ (delLdelH2* outputGate.getResult(1)[0]* (1- Math.pow(TANH.calc(cs.get(1)[0]), 2))));
        assertEquals(0.05019f, delLdelC2_gesamt, DELTA);

        //Backpropagation through Time (BPTT) Ausgangslage (t=2)
        t= 1;       //TimeStep
        //Vorbereitete Werte
        assertEquals(0.5f, inputs[t], DELTA);
        assertEquals(0.2963f, hs.get(0)[0], DELTA);
        assertEquals(0.4437f, cs.get(0)[0], DELTA);
        assertEquals(0.5938f, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.6128f, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.4921f, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.6499f, outputGate.getResult(t)[0], DELTA);
        assertEquals(0.5651f, cs.get(1)[0], DELTA);

        float delF3delH2= (float)(SIGMOID_DERIVATIVE.calc(forgetGate.getResult(2)[0]));
        float delI3delH2= (float)(SIGMOID_DERIVATIVE.calc(inputGate.getResult(2)[0]));

        float delLdelCan3= (delLdelC3*inputGate.getResult(2)[0]);
        float delCan3delH2= (float)(TANH_DERIVATIVE.calc(candidateGate.getResult(2)[0]));
        float delO3delH2= (float)(SIGMOID_DERIVATIVE.calc(outputGate.getResult(2)[0]));
        float delH2= delLdelF3*delF3delH2*u_f + delLdelI3*delI3delH2*u_i + delLdelCan3*delCan3delH2* u_c + delLdelO3*delO3delH2* u_o;
        assertEquals(0.012589f, delH2, DELTA);

        float delLdelC2_direkt= delLdelC3* forgetGate.getResult(2)[0];
        assertEquals(0.04415f, delLdelC2_direkt, DELTA);

        //1. Schritt: Der Gesamtfehler am Cell State delLdelCs2
        delLdelC2_gesamt= (float)(delLdelC2_direkt+(delLdelH2* outputGate.getResult(t)[0]*(1-Math.pow(TANH.calc(cs.get(1)[0]),2))));
        assertEquals(0.05019f, delLdelC2_gesamt, DELTA);

        //Gradienten der Gates

        //Um den Fehler von Zeitschritt 3 zurück nach Zeitschritt 2 zu schieben, nutzen wir die Formel:
        float c3= forgetGate.getResult(t)[0]* cs.get(t-1)[0]+ inputGate.getResult(t)[0]* candidateGate.getResult(t)[0];
        float delC2_total= (float)(delCs3* forgetGate.getResult(t+1)[0]+ delLdelH2* outputGate.getResult(t)[0]* TANH_DERIVATIVE.calc(cs.get(t)[0]));
        assertEquals(0.05019f, delC2_total, DELTA);

        //Alternative Rechnung(Ergebnis ist genauer):
        float direkterPfad= delCs3* forgetGate.getResult(t+1)[0];
        assertEquals(0.04414f, direkterPfad, DELTA);
        float pfadUeberH2= (float)(delLdelH2* outputGate.getResult(t)[0]* (1-Math.pow(TANH.calc(cs.get(t)[0]), 2)));
        assertEquals(0.00604f, pfadUeberH2, DELTA);
        //Gesamtfehler:
        delC2_total= direkterPfad+ pfadUeberH2;
        assertEquals(0.05019f, delC2_total, 1e-5f);

        //Die Gradienten:
        //ForgetGate
        float delF2= delC2_total* cs.get(0)[0];
        assertEquals(0.02228, delF2, 1e-4f);
        //Lokaler Gradient für <u_f> im Zeitschritt t=2
        float delLdelUf2= (float) (delF2* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(1)[0])* hs.get(0)[0]);
        assertEquals(0.00159f, delLdelUf2, 1e-4f);
        //Die Summation (BPTT) Der totale Gradient für das rekurrente Gewicht <u_f>
        delLdelUf_total+= delLdelUf2;
        assertEquals(0.00361f, delLdelUf_total, 1e-4f);
        /*TODO
            Bis hier sind[nach Backpropagation(für Zeitschritt t=2)] der Gradient sowie
            der rekurrente Gradient für das Forget-Gate bestimmt.
            Es fehlen aber noch die Berechnungen der Gradienten der Input-/Candidate-/Output-Gates.
         */
    }

    @Test
    public void testsMultipleTimestepsForwardingAndBackpropagationC() {

/*TODO
        Das Gemini-Bsp (\Desktop\GoogleLstmTestBspBackForwardRechenbeispielMitRekurrentenGewichtenDreiTimesteps)
        enthält NICHT die Anweisung
            "Mach ein konkretes Zahlenbeispiel für den Backpropagation-Schritt mit einem target von 0.5"
        Im Bsp wird ein andere/s Fehler/Target verwendet!
        DESWEGEN stimmen die Werte nicht überein!
*/
        /*An Google(gemini)
            "LSTM: Ein Rechenbeispiel sowohl für Forwarding als auch für Backpropagation, mit Inputs x={1.0, 0.5, 2.0}, rekurrente Gewicht U={Uf=0.1, Ui=0.2, Uc=0.3, Uo=0.4}, Gewicht W={Wf=0.5, Wi=0.6, Wc=0.7, Wo=0.8}, cs[t-1]= 0.0, hs[t-1]=0.0 und Bias b=0.1 wobei ich die Verwendung der rekurrenten Gewichte(Uf,Ui,Uc,Uo) nachvollziehen kann. Ein Rechenbeispiel mit 3 Zeitschritte."
            "Mach ein konkretes Zahlenbeispiel für den Backpropagation-Schritt mit einem target von 0.5"
            "Ich erhalte für h3=0.7083, was habe ich falsch gemacht?"
            "Ich erhalte für c3=1.1618077, was habe ich falsch gemacht?"
            "bitte rechne das nochmal nach mit h3=0.7083 und c3=1.1618077"
         */
        float DELTA = 1e-3f;
        Func SIGMOID = x -> (float) (1 / (1 + Math.exp(-x[0])));
        Func SIGMOID_DERIVATIVE = x -> (x[0] * (1 - x[0]));
        Func TANH = x -> (float) ((Math.exp(x[0]) - Math.exp(-x[0])) / (Math.exp(x[0]) + Math.exp(-x[0])));
        Func TANH_DERIVATIVE = x -> (1.0 - (Math.pow(x[0], 2)));
        float ETA = 0.01f;
        //Google: LSTM: Ein Rechenbeispiel sowohl für Forwarding als auch für Backpropagation, wobei ich die Verwendung der rekurrenten Gewichte(Uf,Ui,Uc,Uo) nachvollziehen kann. Ein Rechenbeispiel mit 3 Zeitschritte.

        float[] inputs = new float[]{1.0f, 0.5f, 2.0f};   //x0, x1, x2
        ArrayList<float[]> cs = new ArrayList<>();
        float[] hs_pre = new float[]{0.0f};
        float[] cs_pre = new float[]{0.0f};
        cs.add(cs_pre);              // DONE: 29.04.2026 es ist notwendig dass ich die HS und CS pro timestep merke => there is no need for <cs_pre> anymore
        ArrayList<float[]> hs = new ArrayList<>();
        hs.add(hs_pre);              // DONE: 29.04.2026 es ist notwendig dass ich die HS und CS pro timestep merke => there is no need for <hs_pre> anymore

        /* TODO: 05.05.2026
         *  So wie es aussieht muss es Gewichte pro Zeitschritt geben.
         *  Denn Update der Gewichte (kumuliert):
         *    Die finalen Gradienten für die Gewichte (z.B. <w_o> oder <w_u>) sind
         *    die Summe der Gradienten aus allen Zeitschritten:
         *      Gradient_w_o[gesamt]= Gradient_w_o[t=2] + Gradient_w_o[t=1] + Gradient_w_o[t=0]
         *  Resultierend wird am Ende der Backpropagation w_*[t] und u_*[t] geupdatet
         *      So dass man sich die (angepassten)Gewichte der Zeitschritte merkt.
         */
        float w_f = 0.5f;
        float w_i = 0.6f;
        float w_c = 0.7f;
        float w_o = 0.8f;
        float u_f = 0.1f;
        float u_i = 0.2f;
        float u_c = 0.3f;
        float u_o = 0.4f;
        float BIAS = 0.1f;

        float[] Xt = new float[hs.get(hs.size() - 1).length + inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            Xt[i] = inputs[i];                           //nach Xt[] werden zuerst die input-Werte aus xt[] geschrieben...
        }

        //Forward Propagation (Timestep 1)
        int t = 0;       //TimeStep
        Gate forgetGate = new Gate(SIGMOID, Xt, new float[][]{{w_f}}, new float[]{BIAS}, u_f, hs.get(hs.size() - 1), t);
        Gate inputGate = new Gate(SIGMOID, Xt, new float[][]{{w_i}}, new float[]{BIAS}, u_i, hs.get(hs.size() - 1), t);
        Gate candidateGate = new Gate(TANH, Xt, new float[][]{{w_c}}, new float[]{BIAS}, u_c, hs.get(hs.size() - 1), t);
        Gate outputGate = new Gate(SIGMOID, Xt, new float[][]{{w_o}}, new float[]{BIAS}, u_o, hs.get(hs.size() - 1), t);

        cs.set(t, calcNewCS(cs.get(hs.size() - 1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.set(t, calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.646, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.668, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.664, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.711, outputGate.getResult(t)[0], DELTA);
        assertEquals(0.444, cs.get(t)[0], DELTA);
        assertEquals(0.296, hs.get(t)[0], DELTA);

        //Forward Propagation (Timestep 2)
        t = 1;       //TimeStep
        forgetGate.calculateResult(u_f, hs.get(t - 1), t);
        inputGate.calculateResult(u_i, hs.get(t - 1), t);
        candidateGate.calculateResult(u_c, hs.get(t - 1), t);
        outputGate.calculateResult(u_o, hs.get(t - 1), t);

        cs.add(calcNewCS(cs.get(t - 1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.add(calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.594, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.6128, inputGate.getResult(t)[0], 1e-4f);
        assertEquals(0.4921, candidateGate.getResult(t)[0], 1e-4f);
        assertEquals(0.6499, outputGate.getResult(t)[0], 1e-4f);
        assertEquals(0.5651, cs.get(t)[0], 1e-4f);
        assertEquals(0.3326, hs.get(t)[0], 1e-4f);

        //Forward Propagation (Timestep 3)
        t = 2;       //TimeStep
        forgetGate.calculateResult(u_f, hs.get(t - 1), t);
        inputGate.calculateResult(u_i, hs.get(t - 1), t);
        candidateGate.calculateResult(u_c, hs.get(t - 1), t);
        outputGate.calculateResult(u_o, hs.get(t - 1), t);

        cs.add(calcNewCS(cs.get(t - 1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.add(calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.7564, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.7968, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.9216, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.8621, outputGate.getResult(t)[0], DELTA);
        assertEquals(1.1618, cs.get(t)[0], DELTA);
        assertEquals(0.7083, hs.get(t)[0], DELTA);

        //Backpropagation t=3
        t=2;
        float y= 0.5f;
        float delLdelH3= hs.get(t)[0]- y;
        assertEquals(0.208, delLdelH3, DELTA);
        //Schritt 1: Output Gate Gradient
        float delLdelO3= (float)(delLdelH3* TANH.calc(cs.get(t)[0]));
        assertEquals(0.171, delLdelO3, DELTA);
        //Gradient für U_o
        float delLdelZo= (float)(delLdelO3* SIGMOID_DERIVATIVE.calc(outputGate.getResult(t)[0]));
        assertEquals(0.0203, delLdelZo, DELTA);
        float delLdelUo= delLdelZo* hs.get(1)[0];
        assertEquals(0.0067, delLdelUo, DELTA);
        //Rückfluss zu h2
        float delLdelH2= delLdelZo*u_o;
        assertEquals(0.0081, delLdelH2, DELTA);
        //Zellzustand
        float delCS3= (float)(delLdelH3* outputGate.getResult(t)[0]* (1- Math.pow(TANH.calc(cs.get(t)[0]), 2)));
        assertEquals(0.058, delCS3, DELTA);
        //Gate-Fehler
        float delF3= (float)(delCS3* cs.get(t-1)[0]* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(t)[0]));
        assertEquals(0.006, delF3, DELTA);
        float delI3= (float)(delCS3* candidateGate.getResult(t)[0]* SIGMOID_DERIVATIVE.calc(inputGate.getResult(t)[0]));
        assertEquals(0.00865, delI3, DELTA);
        float delCand3= (float)(delCS3*inputGate.getResult(t)[0]* (1-Math.pow(candidateGate.getResult(t)[0],2)));
        assertEquals(0.007, delCand3, DELTA);
        //Gradienten(t=3)
        u_o+= delLdelZo* hs.get(t-1)[0];
        assertEquals(0.0067, delLdelZo* hs.get(t-1)[0], DELTA);
        u_f+= delF3* hs.get(t-1)[0];
        assertEquals(0.002, delF3* hs.get(t-1)[0], DELTA);
        u_i+= delI3* hs.get(t-1)[0];
        assertEquals(0.00287, delI3* hs.get(t-1)[0], DELTA);
        u_c+= delCand3* hs.get(t-1)[0];
        assertEquals(0.00232, delCand3* hs.get(t-1)[0], DELTA);

        //Rückfluss
        delLdelH2= delLdelZo*u_o + delF3*u_f + delI3*u_i + delCand3*u_c;
        assertEquals(0.0125, delLdelH2, DELTA);
        float delLdelCS2= delCS3*forgetGate.getResult(t)[0];
        assertEquals(0.0438, delLdelCS2, DELTA);

        //Backpropagation t=2
        t=1;
        //Jetzt wichtig: Gradienten addieren sich!
        assertEquals(0.0125, delLdelH2, DELTA);
        assertEquals(0.0438, delLdelCS2, DELTA);
        //Output Gate
        float delO2= (float)(delLdelH2* TANH.calc(cs.get(t)[0])* SIGMOID_DERIVATIVE.calc(outputGate.getResult(t)[0]));
        assertEquals(0.00146, delO2, DELTA);
        //Zellzustand
        assertEquals(0.0058, (delLdelCS2* cs.get(t)[0]* (inputGate.getResult(t)[0]*(1-inputGate.getResult(t)[0]))), DELTA);

        float delI2= (float)(delLdelCS2* cs.get(t)[0]* SIGMOID_DERIVATIVE.calc(inputGate.getResult(t)[0]));
        assertEquals(0.0058, delI2, DELTA);
        delLdelCS2+= (float)(delLdelH2* outputGate.getResult(t)[0]* (1- Math.pow(TANH.calc(cs.get(t)[0]), 2)));
        assertEquals(0.0498, delLdelCS2, DELTA);
        //Gates
        float delF2= (float)(delLdelCS2* cs.get(t-1)[0]* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(t)[0]));
        assertEquals(0.00533, delF2, 1e-4f);
        float delCand2= (float)(delLdelCS2*inputGate.getResult(t)[0]* (1-Math.pow(cs.get(t)[0],2)));
        assertEquals(0.00922, delCand2, DELTA);


    }

    @Test
    public void testsMultipleTimestepsForwardingAndBackpropagationB() {

/*TODO
        Das Gemini-Bsp (\Desktop\GoogleLstmTestBspBackForwardRechenbeispielMitRekurrentenGewichtenDreiTimesteps)
        enthält NICHT die Anweisung
            "Mach ein konkretes Zahlenbeispiel für den Backpropagation-Schritt mit einem target von 0.5"
        Im Bsp wird ein andere/s Fehler/Target verwendet!
        DESWEGEN stimmen die Werte nicht überein!
*/
        /*An Google(gemini)
            "LSTM: Ein Rechenbeispiel sowohl für Forwarding als auch für Backpropagation, mit Inputs x={1.0, 0.5, 2.0}, rekurrente Gewicht U={Uf=0.1, Ui=0.2, Uc=0.3, Uo=0.4}, Gewicht W={Wf=0.5, Wi=0.6, Wc=0.7, Wo=0.8}, cs[t-1]= 0.0, hs[t-1]=0.0 und Bias b=0.1 wobei ich die Verwendung der rekurrenten Gewichte(Uf,Ui,Uc,Uo) nachvollziehen kann. Ein Rechenbeispiel mit 3 Zeitschritte."
            "Mach ein konkretes Zahlenbeispiel für den Backpropagation-Schritt mit einem target von 0.5"
            "Ich erhalte für h3=0.7083, was habe ich falsch gemacht?"
            "Ich erhalte für c3=1.1618077, was habe ich falsch gemacht?"
            "bitte rechne das nochmal nach mit h3=0.7083 und c3=1.1618077"
         */
        float DELTA= 1e-3f;
        Func SIGMOID= x-> (float)(1 / (1 + Math.exp(-x[0])));
        Func SIGMOID_DERIVATIVE= x->(x[0]*(1-x[0]));
        Func TANH= x-> (float)((Math.exp(x[0])-Math.exp(-x[0])) / (Math.exp(x[0])+Math.exp(-x[0])));
        Func TANH_DERIVATIVE= x->(1.0 - (Math.pow(x[0], 2)));
        float ETA= 0.01f;
        //Google: LSTM: Ein Rechenbeispiel sowohl für Forwarding als auch für Backpropagation, wobei ich die Verwendung der rekurrenten Gewichte(Uf,Ui,Uc,Uo) nachvollziehen kann. Ein Rechenbeispiel mit 3 Zeitschritte.

        float[]inputs= new float[]{1.0f, 0.5f, 2.0f};   //x0, x1, x2
        ArrayList<float[]> cs= new ArrayList<>();
        float[]hs_pre= new float[]{0.0f};
        float[]cs_pre= new float[]{0.0f};
        cs.add(cs_pre);              // DONE: 29.04.2026 es ist notwendig dass ich die HS und CS pro timestep merke => there is no need for <cs_pre> anymore
        ArrayList<float[]> hs= new ArrayList<>();
        hs.add(hs_pre);              // DONE: 29.04.2026 es ist notwendig dass ich die HS und CS pro timestep merke => there is no need for <hs_pre> anymore

        /* TODO: 05.05.2026
         *  So wie es aussieht muss es Gewichte pro Zeitschritt geben.
         *  Denn Update der Gewichte (kumuliert):
         *    Die finalen Gradienten für die Gewichte (z.B. <w_o> oder <w_u>) sind
         *    die Summe der Gradienten aus allen Zeitschritten:
         *      Gradient_w_o[gesamt]= Gradient_w_o[t=2] + Gradient_w_o[t=1] + Gradient_w_o[t=0]
         *  Resultierend wird am Ende der Backpropagation w_*[t] und u_*[t] geupdatet
         *      So dass man sich die (angepassten)Gewichte der Zeitschritte merkt.
         */
        float w_f= 0.5f;
        float w_i= 0.6f;
        float w_c= 0.7f;
        float w_o= 0.8f;
        float u_f= 0.1f;
        float u_i= 0.2f;
        float u_c= 0.3f;
        float u_o= 0.4f;
        float BIAS= 0.1f;

        float[]Xt= new float[hs.get(hs.size()-1).length+ inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            Xt[i] = inputs[i];                           //nach Xt[] werden zuerst die input-Werte aus xt[] geschrieben...
        }

        //Forward Propagation (Timestep 1)
        int t= 0;       //TimeStep
        Gate forgetGate= new Gate(SIGMOID, Xt, new float[][]{{w_f}}, new float[]{BIAS}, u_f, hs.get(hs.size()-1), t);
        Gate inputGate= new Gate(SIGMOID, Xt, new float[][]{{w_i}}, new float[]{BIAS}, u_i, hs.get(hs.size()-1), t);
        Gate candidateGate= new Gate(TANH, Xt, new float[][]{{w_c}}, new float[]{BIAS}, u_c, hs.get(hs.size()-1), t);
        Gate outputGate= new Gate(SIGMOID, Xt, new float[][]{{w_o}}, new float[]{BIAS}, u_o, hs.get(hs.size()-1), t);

        cs.set(t,calcNewCS(cs.get(hs.size()-1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.set(t,calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.646, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.668, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.664, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.711, outputGate.getResult(t)[0], DELTA);
        assertEquals(0.444, cs.get(t)[0], DELTA);
        assertEquals(0.296, hs.get(t)[0], DELTA);

        //Forward Propagation (Timestep 2)
        t= 1;       //TimeStep
        forgetGate.calculateResult(u_f, hs.get(t-1), t);
        inputGate.calculateResult(u_i, hs.get(t-1), t);
        candidateGate.calculateResult(u_c, hs.get(t-1), t);
        outputGate.calculateResult(u_o, hs.get(t-1), t);

        cs.add(calcNewCS(cs.get(t-1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.add(calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.594, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.6128, inputGate.getResult(t)[0], 1e-4f);
        assertEquals(0.4921, candidateGate.getResult(t)[0], 1e-4f);
        assertEquals(0.6499, outputGate.getResult(t)[0], 1e-4f);
        assertEquals(0.5651, cs.get(t)[0], 1e-4f);
        assertEquals(0.3326, hs.get(t)[0], 1e-4f);

        //Forward Propagation (Timestep 3)
        t= 2;       //TimeStep
        forgetGate.calculateResult(u_f, hs.get(t-1), t);
        inputGate.calculateResult(u_i, hs.get(t-1), t);
        candidateGate.calculateResult(u_c, hs.get(t-1), t);
        outputGate.calculateResult(u_o, hs.get(t-1), t);

        cs.add(calcNewCS(cs.get(t-1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.add(calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.7564, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.7968, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.9216, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.8621, outputGate.getResult(t)[0], DELTA);
        assertEquals(1.1618, cs.get(t)[0], DELTA);
        assertEquals(0.7083, hs.get(t)[0], DELTA);

        //Backpropagation through Time (BPTT) Ausgangslage (t=3)
        float TARGET= 0.5f;
        float L= (float)((0.5)*Math.pow(hs.get(t)[0]-TARGET, 2));//Loss nach MSE
        assertEquals(0.0216f, L, DELTA);
        float delLdelH3= hs.get(t)[0]-TARGET;                             //Der Fehlergradient am Output (Delta(h3))
        assertEquals(0.2083f, delLdelH3, DELTA);
        //Zuerst berechnen wir den Einfluss auf das Output Gate:
        TANH.calc(cs.get(t)[0]);                                          //Anteil am Cell State
        float delLdelO3= (float)(delLdelH3* TANH.calc(cs.get(t)[0]));     //Gradient für Output-Gate (o3)
        assertEquals(0.1711f, delLdelO3, DELTA);
        float delO3= delLdelO3* outputGate.getResult(t)[0]* (1-outputGate.getResult(t)[0]);        //Gewichtsupdate Wo
        assertEquals(0.02034f, delO3, DELTA);

        //Gradient für den Cell State (c3)
        //Hier fließt der Fehler von h3 durch die tanh-Funktion in den Zellzustand:
        float delLdelC3= (float)(delLdelH3* outputGate.getResult(t)[0]* (1-(Math.pow(TANH.calc(cs.get(t)[0]),2))));
        assertEquals(0.05837f, delLdelC3, DELTA);

        //Gradienten für Forget- und Input-Gate
        //Jetzt verteilen wir delLdelC3 auf die Gates(Forget Gate, Input Gate), die cs[2] gebildet haben:
        //Forget Gate (f3):
        float delLdelF3= delLdelC3* cs.get(1)[0];
        assertEquals(0.03298f, delLdelF3, DELTA);
        float delF3= (float)(delLdelF3* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(t)[0]));
        assertEquals(0.00608f, delF3, DELTA);
        //Input Gate (i3):
        float delLdelI3= delLdelC3* candidateGate.getResult(t)[0];
        assertEquals(0.05378f, delLdelI3, DELTA);
        float delI3= (float)(delLdelI3* SIGMOID_DERIVATIVE.calc(inputGate.getResult(t)[0]));
        assertEquals(0.00871f, delI3, DELTA);
        //Fehler am Candidate Gate (c[3])
        float delC3= (float)(delLdelC3* inputGate.getResult(2)[0]* TANH_DERIVATIVE.calc(candidateGate.getResult(2)[0]));
        assertEquals(0.007f, delC3, DELTA);
        //Gradient für CellState
        //Fehler am Output
        float delH3= hs.get(t)[0]-TARGET;
        delO3= (float)(delH3* TANH.calc(cs.get(t)[0])* SIGMOID_DERIVATIVE.calc(outputGate.getResult(t)[0]));
        assertEquals(0.02034f, delO3, DELTA);
        float delCs3= (float)(delH3* outputGate.getResult(t)[0]* (1-Math.pow(TANH.calc(cs.get(t)[0]), 2)));
        assertEquals(0.05835f, delCs3, DELTA);

        //Update der Gewichte W und U (für Zeitschritt 3)

        // CHECK: 05.05.2026 Google-gemini fragen ob diese Werte{delLdelWo/f/i/c, delLdelUo/f/i/c} richtig geupdatet sind (für t=3)
        float delLdelWo= delO3* inputs[t];                   //Gradient zum Updaten des Gewichts (Wo)
        assertEquals(0.0406f, delLdelWo, DELTA);
        float delLdelWf= delF3* inputs[t];                   //Gradient zum Updaten des Gewichts (Wf)
        assertEquals(0.0122f, delLdelWf, DELTA);
        float delLdelWi= delI3* inputs[t];                   //Gradient zum Updaten des Gewichts (Wi)
        assertEquals(0.0174f, delLdelWi, DELTA);
        float delLdelWc= delC3* inputs[t];                   //Gradient zum Updaten des Gewichts (Wc)
        assertEquals(0.014f, delLdelWc, DELTA);

        float delLdelUo= delO3* hs.get(1)[0];                   //Gradient zum Updaten des rekurrenten Gewichts (Uo)
        assertEquals(0.00676f, delLdelUo, DELTA);
        float delLdelUf3= delF3* hs.get(1)[0];                   //Gradient zum Updaten des rekurrenten Gewichts (Uf)
        assertEquals(0.00202f, delLdelUf3, DELTA);
        float delLdelUf_total= 0;
        delLdelUf_total+= delLdelUf3;
        float delLdelUi= delI3* hs.get(1)[0];                   //Gradient zum Updaten des rekurrenten Gewichts (Ui)
        assertEquals(0.0029f, delLdelUi, DELTA);
        float delLdelUc= delC3* hs.get(1)[0];                   //Gradient zum Updaten des rekurrenten Gewichts (Uc)
        assertEquals(0.00235f, delLdelUc, DELTA);


        float delLdelC2= delLdelC3* forgetGate.getResult(t)[0];
        assertEquals(0.04414f, delLdelC2, DELTA);
        //Fehlerfluss zum vorigen Hidden State delLdelH2
        /*Das ist der komplexere Teil.
         *Der Hidden State hat im Schritt alle vier Gates beeinflusst (über die rekurrenten Gewichte).
         *Daher summiert sich der Fehler aus allen vier Deltas auf:*/
        float delLdelH2= delF3*u_f + delI3*u_i + delC3*u_c + delO3*u_o;
        assertEquals(0.012589f, delLdelH2, DELTA);
        float delLdelC2_gesamt= (float)(delLdelC2+ (delLdelH2* outputGate.getResult(1)[0]* (1- Math.pow(TANH.calc(cs.get(1)[0]), 2))));
        assertEquals(0.05019f, delLdelC2_gesamt, DELTA);
        /*Zusammenfassend:
            Du siehst, dass der Fehler <delLdelHt> von Schritt zu Schritt
            immer kleiner wird (0,2083 → 0,01258)
         */

        //Backpropagation through Time (BPTT) Ausgangslage (t=2)
        //Start der Berechnung für Zeitschritt t=2
        t= 1;       //TimeStep
        //Vorbereitete Werte
        assertEquals(0.5f, inputs[t], DELTA);
        assertEquals(0.2963f, hs.get(0)[0], DELTA);
        assertEquals(0.4437f, cs.get(0)[0], DELTA);
        assertEquals(0.5938f, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.6128f, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.4921f, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.6499f, outputGate.getResult(t)[0], DELTA);
        assertEquals(0.5651f, cs.get(1)[0], DELTA);

        float delF3delH2= (float)(SIGMOID_DERIVATIVE.calc(forgetGate.getResult(2)[0]));
        float delI3delH2= (float)(SIGMOID_DERIVATIVE.calc(inputGate.getResult(2)[0]));

        float delLdelCan3= (delLdelC3*inputGate.getResult(2)[0]);
        float delCan3delH2= (float)(TANH_DERIVATIVE.calc(candidateGate.getResult(2)[0]));
        float delO3delH2= (float)(SIGMOID_DERIVATIVE.calc(outputGate.getResult(2)[0]));
        float delH2= delLdelF3*delF3delH2*u_f + delLdelI3*delI3delH2*u_i + delLdelCan3*delCan3delH2* u_c + delLdelO3*delO3delH2* u_o;
        assertEquals(0.012589f, delH2, DELTA);

        float delLdelC2_direkt= delLdelC3* forgetGate.getResult(2)[0];
        assertEquals(0.04415f, delLdelC2_direkt, DELTA);

        //1. Schritt: Der Gesamtfehler am Cell State delLdelCs2
        delLdelC2_gesamt= (float)(delLdelC2_direkt+(delLdelH2* outputGate.getResult(t)[0]*(1-Math.pow(TANH.calc(cs.get(1)[0]),2))));
        assertEquals(0.05019f, delLdelC2_gesamt, DELTA);

        //Gradienten der Gates

        /*Der Pfad über den Cell State(cs) ist das Herzstück des LSTMs.
         * Er erlaubt es dem Gradienten, über viele Zeitschritte hinweg zu fließen,
         * ohne sofort zu verschwinden (daher "Long Short-Term Memory").
         */
        //Um den Fehler von Zeitschritt 3 zurück nach Zeitschritt 2 zu schieben, nutzen wir die Formel:
        float c3= forgetGate.getResult(t)[0]* cs.get(t-1)[0]+ inputGate.getResult(t)[0]* candidateGate.getResult(t)[0];
        float delC2_total= (float)(delCs3* forgetGate.getResult(t+1)[0]+ delLdelH2* outputGate.getResult(t)[0]* TANH_DERIVATIVE.calc(cs.get(t)[0]));
        assertEquals(0.05019f, delC2_total, DELTA);

        //Alternative Rechnung(Ergebnis ist genauer):
        float direkterPfad= delCs3* forgetGate.getResult(t+1)[0];
        assertEquals(0.04414f, direkterPfad, DELTA);
        float pfadUeberH2= (float)(delLdelH2* outputGate.getResult(t)[0]* (1-Math.pow(TANH.calc(cs.get(t)[0]), 2)));
        assertEquals(0.00604f, pfadUeberH2, DELTA);
        //Gesamtfehler:
        delC2_total= direkterPfad+ pfadUeberH2;
        assertEquals(0.05019f, delC2_total, 1e-5f);
        //Die Gradienten:
        //ForgetGate
        float delF2= delC2_total* cs.get(0)[0];
        assertEquals(0.02228, delF2, 1e-4f);
        //Lokaler Gradient für <u_f> im Zeitschritt t=2
        float delLdelUf2= (float) (delF2* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(1)[0])* hs.get(0)[0]);
        assertEquals(0.00159f, delLdelUf2, 1e-4f);
        //Die Summation (BPTT) Der totale Gradient für das rekurrente Gewicht <u_f>
        delLdelUf_total+= delLdelUf2;
        assertEquals(0.00361f, delLdelUf_total, 1e-4f);
        /*TODO
            Bis hier sind[nach Backpropagation(für Zeitschritt t=2)] der Gradient sowie
            der rekurrente Gradient für das Forget-Gate bestimmt.
            Es fehlen aber noch die Berechnungen der Gradienten der Input-/Candidate-/Output-Gates.
         */
//-----------------------------------Bis hier hin[oberhalb] stimmts------------------------------------------------------------
        //InputtGate
        float delI2= (float)(delC2_total* candidateGate.getResult(t)[0]* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(t)[0]));
        assertEquals(0.00586f, delI2, 1e-4f);
        //CandidateGate
        float delCan2= (float)(delC2_total* inputGate.getResult(t)[0]* TANH_DERIVATIVE.calc(candidateGate.getResult(t)[0]));
        assertEquals(0.0233f, delCan2, 1e-4f);
        //OutputGate
        float delO2= (float)(delLdelH2* TANH.calc(cs.get(t)[0])* SIGMOID_DERIVATIVE.calc(outputGate.getResult(t)[0]));
        assertEquals(0.00146f, delO2, 1e-4f);
        //Die Gradienten der Gewichte:
        //ForgetGate

        //InputGate

        //CandidateGate

        //OutputGate

        //Die Gradienten der rekurrenten Gewichte:
        //ForgetGate

        //InputGate

        //CandidateGate

        //OutputGate

        //Um den Fehler weiter nach vorne zu tragen, berechnen wir wieder den neuen Hidden-Fehler <delH1>
        float dh1=delF2*u_f + delI2*u_i + delCan2*u_c + delO2*u_o;//:= "delH1"
        assertEquals(0.00928f, dh1, 1e-4f);
        float dc1= delC2_total* forgetGate.getResult(t)[0];
        assertEquals(0.02979f, dc1, 1e-4f);
        //Vorbereitete Werte (aus Vorwärtsgang (t=1))
        t=0;
        assertEquals(1.f, inputs[t], DELTA);
        assertEquals(0.f, hs_pre[0], DELTA);
        assertEquals(0.f, cs_pre[0], DELTA);
        assertEquals(0.6457f, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.6682f, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.664f, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.7109f, outputGate.getResult(t)[0], DELTA);
        assertEquals(0.4437f, cs.get(t)[0], DELTA);

        //Gradienten der Gates
        //Gesamtfehler:
        float delC1_total= (float)(dh1* outputGate.getResult(t)[0]* (1-Math.pow(TANH.calc(cs.get(t)[0]),2))+ dc1);
        assertEquals(0.03524f, delC1_total, 1e-4f);
        //Die Gradienten:
        //ForgetGate
        float delF1= (float)(delC1_total* cs_pre[0]* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(t)[0]));
        assertEquals(0.0f, delF1, 1e-4f);
        //InputtGate
        float delI1= (float)(delC1_total* candidateGate.getResult(t)[0]* SIGMOID_DERIVATIVE.calc(inputGate.getResult(t)[0]));
        assertEquals(0.005187f, delI1, 1e-4f);
        //CandidateGate
        float delCan1= (float)(delC1_total* inputGate.getResult(t)[0]* TANH_DERIVATIVE.calc(candidateGate.getResult(t)[0]));
        assertEquals(0.01317f, delCan1, 1e-4f);
        //OutputGate
        float delC1= delC1_total;
        float delLdelH1= (float)(delC1/(outputGate.getResult(0)[0]* (1-Math.pow(TANH.calc(cs.get(t)[0]),2))));
        assertEquals(0.06f, delLdelH1, 1e-4f);

        //TODO
        //Die Gradienten der Gewichte:
        //ForgetGate
        float delLdelWf1= delF2* inputs[t];
//            assertEquals(0.0f, delLdelWf1, 1e-4f);
        //InputGate
        float delLdelWi1= delI2* inputs[t];
//            assertEquals(0.0052f, delLdelWi1, 1e-4f);
        //CandidateGate
        float delLdelWcan1= delCan2* inputs[t];
//            assertEquals(0.01165f, delLdelWcan1, 1e-4f);
        //OutputGate
        float delLdelWo1= delO2* inputs[t];
//            assertEquals(0.00073f, delLdelWo1, 1e-4f);
        //Die Gradienten der rekurrenten Gewichte:
        //ForgetGate
        float delLdelUf1= delF2* hs.get(0)[0];
        assertEquals(0.00159f, delLdelUf1, 1e-4f);
        //InputGate
        float delLdelUi1= delI2* hs.get(0)[0];
        assertEquals(0.00174f, delLdelUi1, 1e-4f);
        //CandidateGate
        float delLdelUcan1= delCan2* hs.get(0)[0];
        assertEquals(0.0069f, delLdelUcan1, 1e-4f);
        //OutputGate
        float delLdelUo1= delO2* hs.get(0)[0];
        assertEquals(0.00043f, delLdelUo1, 1e-4f);

//----------------------------------------------------------------------------------------------------------------------

        //Ich habe jetzt die beiden entscheidenden Fehlerwerte, um die Gewichte in Zeitschritt 2 anzupassen:
        assertEquals(0.012586f, delLdelH2, DELTA);  //Fehler, der durch die rekurrenten Gewichte <U> zurückkam
        assertEquals(0.05018f, delC2_total, DELTA);   //Fehler, der durch das "Gedächtnis" zurückkam
        /* Mit diesen Werten könntest du jetzt die Gradienten für W, U und b im zweiten Zeitschritt berechnen,
         *  genau wie wir es für Schritt 3 getan haben. */

        /*Um den Gradienten für das Gewicht <w_f>(das Gewicht des Inputs für das Forget-Gate)
         * im Zeitschritt 2 zu berechnen, nutzen wir den lokalen Fehler am Forget-Gate dieses Schritts.
         */
        t=1;
        //1. Benötigte Werte für Zeitschritt 2
        assertEquals(0.5f, inputs[t], DELTA);
        assertEquals(0.05018f, delC2_total, DELTA);
        assertEquals(0.4442f, cs.get(t-1)[0], DELTA);
        assertEquals(0.594f, forgetGate.getResult(t)[0], DELTA);
        //2. Lokaler Fehler am Forget-Gate(delF2_pre)
        float delF2_pre= (float)(delC2_total* cs.get(t-1)[0]* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(t)[0]));
        assertEquals(0.00537f, delF2_pre, DELTA);
        //3.Gradient für <w_f>
        delLdelWf= delF2_pre* inputs[t];    //Dieser Wert sagt uns, wie das Gewicht für den externen Input angepasst werden muss.
        assertEquals(0.002685f, delLdelWf, DELTA);

        //Gradienten für das rekurrente Gewicht(<u_f>)
        //Benötigte Werte für Zeitschritt 2
        assertEquals(0.00537f, delF2_pre, DELTA);
        assertEquals(0.2964f, hs.get(0)[0], DELTA);
        //Gradient für <u_f> (Schritt 2)
        delLdelUf2= delF2_pre* hs.get(0)[0]; //Dieser Wert sagt uns, wie die Verbindung zum Gedächtnis (der rekurrente Teil) angepasst werden muss.
        assertEquals(0.00159f, delLdelUf2, DELTA);

        float delLdelBiasfTotal= delF2_pre+ delF3;
        assertEquals(0.01145f, delLdelBiasfTotal, DELTA);

        //Input Gate(i2)
        assertEquals(0.4921f, candidateGate.getResult(1)[0], DELTA);
        float delLdelI2= delLdelC2* candidateGate.getResult(1)[0];
        assertEquals(0.02171f, delLdelI2, DELTA);

        //Cell Candidate(c2)
        assertEquals(0.6128f, inputGate.getResult(1)[0], DELTA);
        float delLdelCan2= delLdelC2* inputGate.getResult(1)[0];
        assertEquals(0.02703f, delLdelCan2, DELTA);

        //Forget Candidate(f2)
        assertEquals(0.6128f, inputGate.getResult(1)[0], DELTA);
        float delLdelF2= delLdelC2* cs.get(0)[0];
        assertEquals(0.01957f, delLdelF2, DELTA);

        //Umwandlung in die "Pre-Activation" Gradienten <delta>
        /*Um die Gewichte (W, U) anzupassen,
         *müssen wir den Fehler noch durch die Ableitung der Aktivierungsfunktionen sig oder tanh schicken:
         */
        //Für das Input Gate:
        delI2= (float)(delLdelI2* SIGMOID_DERIVATIVE.calc(inputGate.getResult(1)[0]));
        assertEquals(0.00515f, delI2, 1e-5f);
        //Für den Cell Candidate:
        delCan2= (float)(delLdelCan2* TANH_DERIVATIVE.calc(candidateGate.getResult(1)[0]));
        assertEquals(0.02048f, delCan2, 1e-4f);
        //Für den Forget Gate:
        delF2= (float)(delLdelF2* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(1)[0]));
        assertEquals(0.00472f, delF2, 1e-5f);
        //Für das Output Gate:
        delO2= (float)(delLdelH2* TANH.calc(cs.get(1)[0])* SIGMOID_DERIVATIVE.calc(outputGate.getResult(1)[0]));
        assertEquals(0.00116f, delO2, DELTA);

        //Gewichts-Update von <w_i> in Schritt 2:
        //Wenn wir das Gewicht <w_i> (für den Input <x2> = 0.5 in diesem Schritt korrigieren wollen:
//        delLdelWi2= delI2* inputs[1];
//        assertEquals(0.002575f, delLdelWi2, 1e-5f);
        //Gewichts-Update von <w_c> in Schritt 2:
        //Wenn wir das Gewicht <w_c> (für den Input <x2> = 0.5 in diesem Schritt korrigieren wollen:
        float delLdelWc2= delCan2* inputs[1];
        assertEquals(0.0102512725f, delLdelWc2, 1e-5f);


        /* Um die Akkumulation für das Gewicht Wc (Zellkandidat-Gewicht) abzuschließen,
         * fassen wir die Gradienten aus allen drei Zeitschritten zusammen.
         */
        //Zeitschritt t=3(x[3]=2.0)
        //Aus unseren ersten Rechnungen für den Output-Fehler (h3-y = 0.2083)):
        float delCan3= (float)(delLdelC3* inputGate.getResult(2)[0]* TANH_DERIVATIVE.calc(candidateGate.getResult(2)[0]));
        assertEquals(0.007f, delCan3, 1e-5f);
        float delWc3= delCan3* inputs[2];
        assertEquals(0.014f, delWc3, 1e-5f);

        //Zeitschritt t=2(x[2]=0.5)
        //Aus unseren ersten Rechnungen für den Output-Fehler (h3-y = 0.2083)):
//        float delWc2= delCan3* inputs[2];delLdelWc2
        assertEquals(0.01025f, delLdelWc2, 1e-5f);

        //Update für das Forget-Gate Wf in Schritt 2
        assertEquals(0.04411f, delLdelC2, 1e-4f);

        delF2= (float)(delLdelC2* cs.get(0)[0]* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(1)[0]));
        assertEquals(0.00472f, delF2, 1e-5f);
        float delWf2= delF2* inputs[1];
        assertEquals(0.00236f, delWf2, 1e-5f);

        //Update für das Output-Gate Wo in Schritt 2
        assertEquals(0.00146f, delO2, 1e-5f);
        float delWo2= delO2* inputs[1];
        assertEquals(0.00073f, delWo2, 1e-5f);

        //Zeitschritt t=1(x[1]=1.0)
        //Aus unseren ersten Rechnungen für den Output-Fehler (h3-y = 0.2083)):
        float delLdelC1= (float)(delLdelC2* forgetGate.getResult(1)[0]);
        assertEquals(0.02619f, delLdelC1, DELTA);
        delCan1= (float)(delLdelC1* inputGate.getResult(0)[0]* TANH_DERIVATIVE.calc(candidateGate.getResult(0)[0]));
        assertEquals(0.00978f, delCan1, 1e-4f);
        float delWc1= delCan1* inputs[0];

        //4. Das finale Update
        //Nun addieren wir alle Gradienten auf:
        float delLdelWc_gesamt= delWc3 + delLdelWc2 + delWc1;
        assertEquals(0.03403f, delLdelWc_gesamt, 1e-4f);
/*TODO
        Das Gemini-Bsp (\Desktop\GoogleLstmTestBspBackForwardRechenbeispielMitRekurrentenGewichtenDreiTimesteps)
        enthält NICHT die Anweisung
               "Mach ein konkretes Zahlenbeispiel für den Backpropagation-Schritt mit einem target von 0.5"
        Im Bsp wird ein andere/s Fehler/Target verwendet!
        DESWEGEN übereinstimmen die Werte nicht!
*/
    }
    @Test
    public void testsMultipleTimestepsForwardingAndBackpropagationA() {
        /*An Google(gemini)
            "LSTM: Ein Rechenbeispiel sowohl für Forwarding als auch für Backpropagation, mit Inputs x={1.0, 0.5, 2.0}, rekurrente Gewicht U={Uf=0.1, Ui=0.2, Uc=0.3, Uo=0.4}, Gewicht W={Wf=0.5, Wi=0.6, Wc=0.7, Wo=0.8}, cs[t-1]= 0.0, hs[t-1]=0.0 und Bias b=0.1 wobei ich die Verwendung der rekurrenten Gewichte(Uf,Ui,Uc,Uo) nachvollziehen kann. Ein Rechenbeispiel mit 3 Zeitschritte."
            "Mach ein konkretes Zahlenbeispiel für den Backpropagation-Schritt mit einem target von 0.5"
            "Wie berechne ich den Gradienten für Bias mit Rechenbeispiel? Wie update ich den Bias?"
         */
        float DELTA= 1e-3f;
        Func SIGMOID= x-> (float)(1 / (1 + Math.exp(-x[0])));
        Func SIGMOID_DERIVATIVE= x->(x[0]*(1-x[0]));
        Func TANH= x-> (float)((Math.exp(x[0])-Math.exp(-x[0])) / (Math.exp(x[0])+Math.exp(-x[0])));
        // TODO: 30.04.2026 REPAIRRRRRR Func TANH_DERIVATIVE= x->(1.0 - (Math.pow(x[0], 2)));
        //Google: LSTM: Ein Rechenbeispiel sowohl für Forwarding als auch für Backpropagation, wobei ich die Verwendung der rekurrenten Gewichte(Uf,Ui,Uc,Uo) nachvollziehen kann. Ein Rechenbeispiel mit 3 Zeitschritte.

        float[]inputs= new float[]{1.0f, 0.5f, 2.0f};   //x0, x1, x2
        ArrayList<float[]> cs= new ArrayList<>();
        float[]hs_pre= new float[]{0.0f};
        float[]cs_pre= new float[]{0.0f};
        cs.add(cs_pre);              // DONE: 29.04.2026 es ist notwendig dass ich die HS und CS pro timestep merke => there is no need for <cs_pre> anymore
        ArrayList<float[]> hs= new ArrayList<>();
        hs.add(hs_pre);              // DONE: 29.04.2026 es ist notwendig dass ich die HS und CS pro timestep merke => there is no need for <hs_pre> anymore
        float w_f= 0.5f;
        float w_i= 0.6f;
        float w_c= 0.7f;
        float w_o= 0.8f;
        float u_f= 0.1f;
        float u_i= 0.2f;
        float u_c= 0.3f;
        float u_o= 0.4f;
        float BIAS= 0.1f;

        float[]Xt= new float[hs.get(hs.size()-1).length+ inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            Xt[i] = inputs[i];                           //nach Xt[] werden zuerst die input-Werte aus xt[] geschrieben...
        }

        //Forward Propagation (Timestep 1)
        int t= 0;       //TimeStep
        Gate forgetGate= new Gate(SIGMOID, Xt, new float[][]{{w_f}}, new float[]{BIAS}, u_f, hs.get(hs.size()-1), t);
        Gate inputGate= new Gate(SIGMOID, Xt, new float[][]{{w_i}}, new float[]{BIAS}, u_i, hs.get(hs.size()-1), t);
        Gate candidateGate= new Gate(TANH, Xt, new float[][]{{w_c}}, new float[]{BIAS}, u_c, hs.get(hs.size()-1), t);
        Gate outputGate= new Gate(SIGMOID, Xt, new float[][]{{w_o}}, new float[]{BIAS}, u_o, hs.get(hs.size()-1), t);

        cs.set(t,calcNewCS(cs.get(hs.size()-1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.set(t,calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.646, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.668, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.664, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.711, outputGate.getResult(t)[0], DELTA);
        assertEquals(0.444, cs.get(t)[0], DELTA);
        assertEquals(0.296, hs.get(t)[0], DELTA);

        //Forward Propagation (Timestep 2)
        t= 1;       //TimeStep
        forgetGate.calculateResult(u_f, hs.get(t-1), t);
        inputGate.calculateResult(u_i, hs.get(t-1), t);
        candidateGate.calculateResult(u_c, hs.get(t-1), t);
        outputGate.calculateResult(u_o, hs.get(t-1), t);

        cs.add(calcNewCS(cs.get(t-1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.add(calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.594, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.6128, inputGate.getResult(t)[0], 1e-4f);
        assertEquals(0.4921, candidateGate.getResult(t)[0], 1e-4f);
        assertEquals(0.6499, outputGate.getResult(t)[0], 1e-4f);
        assertEquals(0.5651, cs.get(t)[0], 1e-4f);
        assertEquals(0.3326, hs.get(t)[0], 1e-4f);

        //Forward Propagation (Timestep 3)
        t= 2;       //TimeStep
        forgetGate.calculateResult(u_f, hs.get(t-1), t);
        inputGate.calculateResult(u_i, hs.get(t-1), t);
        candidateGate.calculateResult(u_c, hs.get(t-1), t);
        outputGate.calculateResult(u_o, hs.get(t-1), t);

        cs.add(calcNewCS(cs.get(t-1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.add(calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.7564, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.7968, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.9216, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.8621, outputGate.getResult(t)[0], DELTA);
        assertEquals(1.1618, cs.get(t)[0], DELTA);
        assertEquals(0.7083, hs.get(t)[0], DELTA);

        //An Google: "Mach ein konkretes Zahlenbeispiel für den Backpropagation-Schritt mit einem target von 0.5"
        //Um das Beispiel konkret zu machen, nutzen wir den Mean Squared Error (MSE) als Loss-Funktion:
        float target= 0.5f;
        float L= (0.5f)* (float)(Math.pow((hs.get(2)[0]- target), 2));
        assertEquals(0.0218, L, 1e-4f);

        //1. Initialer Fehler am Output (t=3)
        //Zuerst berechnen wir den Gradienten des Loss bezüglich des letzten Hidden States:
        float delLdelH3= hs.get(2)[0]- target;
        assertEquals(0.209, delLdelH3, DELTA);

        //2. Gradientenfluss durch die Gates (t=3)
        //Um die Gewichte anzupassen, müssen wir den Fehler von H3 bzw. <hs.get(2)> zurück in die einzelnen Gates leiten.
        //Fehler am Output Gate (o[3])
        float delO3= delLdelH3* (float)TANH.calc(cs.get(2)[0])* (float)SIGMOID_DERIVATIVE.calc(outputGate.getResult(t)[0]);
        assertEquals(0.0204, delO3, 1e-4f);
        //Fehler am Cell State (cs[3])
        float delLdelC3= (float)(delLdelH3* outputGate.getResult(t)[0]* (1- Math.pow(TANH.calc(cs.get(2)[0]), 2)));
        assertEquals(0.0583, delLdelC3, 1e-4f);
        //Fehler am Forget Gate (f[3])
        float delF3= (float)(delLdelC3* cs.get(1)[0]* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(2)[0]));
        assertEquals(0.0061, delF3, 1e-4f);
        //Fehler am Input Gate (i[3])
        float delI3= (float)(delLdelC3* candidateGate.getResult(2)[0]* SIGMOID_DERIVATIVE.calc(inputGate.getResult(2)[0]));
        assertEquals(0.0087, delI3, 1e-4f);
        //Fehler am Candidate Gate (c[3])
        float delC3= (float)(delLdelC3* inputGate.getResult(2)[0]* (1-Math.pow(candidateGate.getResult(2)[0], 2)));
        assertEquals(0.007, delC3, 1e-4f);

        //3. Gewichtsanpassung (Beispiel <w_f> und <u_f>)
        //Nun berechnen wir die Gradienten für die Gewichte des Forget-Gates im Zeitschritt 3.
        //Für das Input-Gewicht <w_f>:
        float delLdelWf3= delF3* inputs[2];
        assertEquals(0.0122, delLdelWf3, 1e-4f);
        //Für das rekurrente Gewicht <u_f>:
        float delLdelUf3= delF3* hs.get(1)[0];
        assertEquals(0.0020, delLdelUf3, 1e-4f);

        //3. Gewichtsanpassung (Beispiel <w_i> und <u_i>)
        //Nun berechnen wir die Gradienten für die Gewichte des Input-Gates im Zeitschritt 3.
        //Für das Input-Gewicht <w_i>:
        float delLdelWi3= delI3* inputs[2];
        assertEquals(0.0174, delLdelWi3, 1e-4f);
        //Für das rekurrente Gewicht <u_i>:
        float delLdelUi3= delI3* hs.get(1)[0];
        assertEquals(0.0029, delLdelUi3, 1e-4f);

        //3. Gewichtsanpassung (Beispiel <w_c> und <u_c>)
        //Nun berechnen wir die Gradienten für die Gewichte des Candidate-Gates im Zeitschritt 3.
        //Für das Input-Gewicht <w_c>:
        float delLdelWc3= delC3* inputs[2];
        assertEquals(0.0140, delLdelWc3, 1e-4f);
        //Für das rekurrente Gewicht <u_c>:
        float delLdelUc3= delC3* hs.get(1)[0];
        assertEquals(0.0023, delLdelUc3, 1e-4f);

        //3. Gewichtsanpassung (Beispiel <w_o> und <u_o>)
        //Nun berechnen wir die Gradienten für die Gewichte des Candidate-Gates im Zeitschritt 3.
        //Für das Input-Gewicht <w_o>:
        float delLdelWo3= delO3* inputs[2];
        assertEquals(0.0408, delLdelWo3, DELTA);
        //Für das rekurrente Gewicht <u_o>:
        float delLdelUo3= delO3* hs.get(1)[0];
        assertEquals(0.0068, delLdelUo3, 1e-4f);

        //4. Backpropagation zu Zeitschritt 2 (t=2)
        //1. Der Fehlerfluss zum Hidden State(<delLdelHS2>)
        //Um h2 zu berechnen, summiert man die Einflüsse aller Gates aus Schritt 3, die U verwenden:
        float delLdelHS2= (delF3* u_f)+ (delI3* u_i)+ (delC3* u_c)+ (delO3* u_o);
        assertEquals(0.01261, delLdelHS2, 1e-4f);

        //2. Der Fehlerfluss zum Cell State (Gesamtfehler_fuer_CS2)
        //Dies ist die wichtigste Rechnung des LSTMs. Der Fehler an speist sich aus zwei Quellen:
        //Quelle A: Der direkte Rückfluss durch die "Zell-Autobahn"
        //Der Fehler von CS3 fließt direkt durch das Forget Gate f3 zurück.
        float Der_Fehler_von_CS3= delLdelC3* forgetGate.getResult(2)[0];
        assertEquals(0.04414, Der_Fehler_von_CS3, DELTA);
        //Quelle B: Der Rückfluss über den aktuellen Hidden State hs2
        //Da HS2 direkt von CS2 abhängt (HS2= o2* tanh(CS2)), muss der Fehler von HS2(aus Punkt 1 oben) durch
        // das Output Gate und die Ableitung von Tanh zurückgerechnet werden.
        float Der_Fehler_von_H2_durch_O2=
                (float)(delLdelHS2* outputGate.getResult(1)[0]* (1- Math.pow(TANH.calc(cs.get(1)[0]), 2)));
        assertEquals(0.00604, Der_Fehler_von_H2_durch_O2, DELTA);
        //Das Ergebnis: Die Summe beider Quellen ergibt den Gesamtfehler für den Zellzustand t=2:
        float Gesamtfehler_fuer_CS2= Der_Fehler_von_CS3+ Der_Fehler_von_H2_durch_O2;
        assertEquals(0.05011, Gesamtfehler_fuer_CS2, 1e-4f);

        //3. Lokale Gate-Fehler in Zeitschritt t=2 berechnen
        //Mit nun bekannten <Gesamtfehler_fuer_CS2> und <delLdelHS2> berechnen wir die Gradienten der Gewichte in diesem Schritt.
        //Fehler am Forget Gate (f[2])
        float delF2= (float)(Gesamtfehler_fuer_CS2* cs.get(0)[0]* SIGMOID_DERIVATIVE.calc(forgetGate.getResult(1)[0]));
        assertEquals(0.00537, delF2, 1e-5f);
        //Fehler am Output Gate (o[2])
        float delO2= delLdelHS2* (float)TANH.calc(cs.get(1)[0])* (float)SIGMOID_DERIVATIVE.calc(outputGate.getResult(1)[0]);
        assertEquals(0.00147, delO2, 1e-4f);
        //Fehler am Cell State (cs[2])
        float delLdelC2= (float)((delLdelC3* forgetGate.getResult(2)[0])+ delLdelHS2* outputGate.getResult(1)[0]* (1- Math.pow(TANH.calc(cs.get(1)[0]), 2)));
        assertEquals(0.05011, delLdelC2, 1e-4f);
        //Fehler am Input Gate (i[2])
        float delI2= (float)(delLdelC2* candidateGate.getResult(1)[0]* SIGMOID_DERIVATIVE.calc(inputGate.getResult(1)[0]));
        assertEquals(0.00585, delI2, 1e-4f);
        //Fehler am Candidate Gate (c[2])
        float delC2= (float)(delLdelC2* inputGate.getResult(1)[0]* (1-Math.pow(candidateGate.getResult(1)[0], 2)));
        assertEquals(0.02328, delC2, 1e-4f);

        //4. Gewichtsanpassung (Beispiel <w_f> und <u_f>)
        //Nun berechnen wir die Gradienten für die Gewichte des Forget-Gates im Zeitschritt 2.
        //Für das Input-Gewicht <w_f>:
        float delLdelWf2= delF2* inputs[1];
        assertEquals(0.00269, delLdelWf2, 1e-5f);
        //Für das rekurrente Gewicht <u_f>:
        float delLdelUf= delF2* hs.get(0)[0];
        assertEquals(0.00159, delLdelUf, 1e-5f);
        //4. Gewichtsanpassung (Beispiel <w_i> und <u_i>)
        //Nun berechnen wir die Gradienten für die Gewichte des Input-Gates im Zeitschritt 2.
        //Für das Input-Gewicht <w_i>:
        float delLdelWi2= delI2* inputs[1];
        assertEquals(0.00293, delLdelWi2, 1e-5f);
        //Für das rekurrente Gewicht <u_i>:
        float delLdelUi2= delI2* hs.get(0)[0];
        assertEquals(0.00173, delLdelUi2, 1e-5f);
        //4. Gewichtsanpassung (Beispiel <w_c> und <u_c>)
        //Nun berechnen wir die Gradienten für die Gewichte des Candidate-Gates im Zeitschritt 2.
        //Für das Input-Gewicht <w_c>:
        float delLdelWc2= delC2* inputs[1];
        assertEquals(0.01164, delLdelWc2, DELTA);
        //Für das rekurrente Gewicht <u_c>:
        float delLdelUc2= delC2* hs.get(0)[0];
        assertEquals(0.00689, delLdelUc2, 1e-4f);
        //4. Gewichtsanpassung (Beispiel <w_o> und <u_o>)
        //Nun berechnen wir die Gradienten für die Gewichte des Output-Gates im Zeitschritt 2.
        //Für das Input-Gewicht <w_c>:
        float delLdelWo2= delO2* inputs[1];
        assertEquals(0.00074, delLdelWo2, 1e-5f);
        //Für das rekurrente Gewicht <u_c>:
        float delLdelUo2= delO2* hs.get(0)[0];
        assertEquals(0.00044, delLdelUo2, 1e-5f);

        //Die Gewichte werden erst angepasst, nachdem die Backpropagation aller Zeitschritte durchgeführt wurde.
        /*Akkumulation: Während du von Schritt 3 zu Schritt 2 und dann zu Schritt 1 zurückgehst,
         * berechnest du für jeden Zeitschritt die sogenannten lokalen Gradienten für jedes Gewicht.*/
        /*Konstante Gewichte: Du musst für die Berechnungen in Schritt 2 und Schritt 1 genau dieselben
         *Gewichtswerte <W> und <U> verwenden, die du auch beim Forward Pass benutzt hast.
         * Würdest du sie zwischendurch ändern, würden die Ableitungen nicht mehr zum ursprünglichen Rechenweg passen.*/
        /*Das Update: Erst wenn du ganz am Anfang (Schritt 1) angekommen bist und die Gradienten aller
         * Zeitschritte für ein Gewicht addiert hast, wird das Gewicht angepasst.*/
    }

    @Test
    public void testsMultipleTimestepsToFixForwardingCellState() {
        /*An Google(gemini)
            "LSTM: Ein Rechenbeispiel sowohl für Forwarding als auch für Backpropagation, mit Inputs x={1.0, 0.5, 2.0}, rekurrente Gewicht U={Uf=0.1, Ui=0.2, Uc=0.3, Uo=0.4}, Gewicht W={Wf=0.5, Wi=0.6, Wc=0.7, Wo=0.8}, cs[t-1]= 0.0, hs[t-1]=0.0 und Bias b=0.1 wobei ich die Verwendung der rekurrenten Gewichte(Uf,Ui,Uc,Uo) nachvollziehen kann. Ein Rechenbeispiel mit 3 Zeitschritte."
            "Mach ein konkretes Zahlenbeispiel für den Backpropagation-Schritt mit einem target von 0.5"
         */
        float DELTA= 1e-3f;
        Func SIGMOID= x-> (float)(1 / (1 + Math.exp(-x[0])));
        Func SIGMOID_DERIVATIVE= x->(x[0]*(1-x[0]));
        Func TANH= x-> (float)((Math.exp(x[0])-Math.exp(-x[0])) / (Math.exp(x[0])+Math.exp(-x[0])));
        //Google: LSTM: Ein Rechenbeispiel sowohl für Forwarding als auch für Backpropagation, wobei ich die Verwendung der rekurrenten Gewichte(Uf,Ui,Uc,Uo) nachvollziehen kann. Ein Rechenbeispiel mit 3 Zeitschritte.

        float[]inputs= new float[]{1.0f, 0.5f, 2.0f};   //x0, x1, x2
        ArrayList<float[]> cs= new ArrayList<>();
        float[]hs_pre= new float[]{0.0f};
        float[]cs_pre= new float[]{0.0f};
        cs.add(cs_pre);              // DONE: 29.04.2026 es ist notwendig dass ich die HS und CS pro timestep merke => there is no need for <cs_pre> anymore
        ArrayList<float[]> hs= new ArrayList<>();
        hs.add(hs_pre);              // DONE: 29.04.2026 es ist notwendig dass ich die HS und CS pro timestep merke => there is no need for <hs_pre> anymore
        float w_f= 0.5f;
        float w_i= 0.6f;
        float w_c= 0.7f;
        float w_o= 0.8f;
        float u_f= 0.1f;
        float u_i= 0.2f;
        float u_c= 0.3f;
        float u_o= 0.4f;
        float BIAS= 0.1f;

        float[]Xt= new float[hs.get(hs.size()-1).length+ inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            Xt[i] = inputs[i];                           //nach Xt[] werden zuerst die input-Werte aus xt[] geschrieben...
        }

        //Forward Propagation (Timestep 1)
        int t= 0;       //TimeStep
        Gate forgetGate= new Gate(SIGMOID, Xt, new float[][]{{w_f}}, new float[]{BIAS}, u_f, hs.get(hs.size()-1), t);
        Gate inputGate= new Gate(SIGMOID, Xt, new float[][]{{w_i}}, new float[]{BIAS}, u_i, hs.get(hs.size()-1), t);
        Gate candidateGate= new Gate(TANH, Xt, new float[][]{{w_c}}, new float[]{BIAS}, u_c, hs.get(hs.size()-1), t);
        Gate outputGate= new Gate(SIGMOID, Xt, new float[][]{{w_o}}, new float[]{BIAS}, u_o, hs.get(hs.size()-1), t);

        //NOTE: 30.04.2026 Hier werden die ersten-/initial- Werte aus cs[] und hs[] überschrieben, das ist nicht zielführend
        cs.set(t,calcNewCS(cs.get(hs.size()-1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.set(t,calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.646, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.668, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.664, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.711, outputGate.getResult(t)[0], DELTA);
        assertEquals(0.444, cs.get(t)[0], DELTA);
        assertEquals(0.296, hs.get(t)[0], DELTA);

        //Forward Propagation (Timestep 2)
        t= 1;       //TimeStep
        forgetGate.calculateResult(u_f, hs.get(t-1), t);
        inputGate.calculateResult(u_i, hs.get(t-1), t);
        candidateGate.calculateResult(u_c, hs.get(t-1), t);
        outputGate.calculateResult(u_o, hs.get(t-1), t);

        cs.add(calcNewCS(cs.get(t-1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.add(calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.594, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.6128, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.4921, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.6499, outputGate.getResult(t)[0], DELTA);
        assertEquals(0.5651, cs.get(t)[0], DELTA);
        assertEquals(0.3326, hs.get(t)[0], DELTA);

        //Forward Propagation (Timestep 3)
        t= 2;       //TimeStep
        forgetGate.calculateResult(u_f, hs.get(t-1), t);
        inputGate.calculateResult(u_i, hs.get(t-1), t);
        candidateGate.calculateResult(u_c, hs.get(t-1), t);
        outputGate.calculateResult(u_o, hs.get(t-1), t);

        cs.add(calcNewCS(cs.get(t-1), forgetGate.getResult(t), inputGate.getResult(t), candidateGate.getResult(t)));
        hs.add(calcNewHS(outputGate.getResult(t), cs.get(t)));

        assertEquals(0.7564, forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.7968, inputGate.getResult(t)[0], DELTA);
        assertEquals(0.9216, candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.8621, outputGate.getResult(t)[0], DELTA);
        assertEquals(1.1618, cs.get(t)[0], DELTA);
        assertEquals(0.7083, hs.get(t)[0], DELTA);
    }

    private float[] calcNewHS(float[] outputGate, float[]cs) {  //Helper for testsMultipleTimestepsToFixForwardingCellState
        //Hadamard-Produkt {outputGate, tanh(cs)}
        float[] result= new float[outputGate.length];
        for(int i=0; i< result.length; i++) {
            result[i]= outputGate[i]* (float)TANH.calc(cs[i]);
        }
        return result;
    }

    //Helper for testsMultipleTimestepsToFixForwardingCellState
    private float[] calcNewCS(float[] csPre, float[] forgetGate, float[] inputGate, float[] candidateGate) {
        //Hadamard-Produkt {csPre, forgetGate}
        float[] hadamardProduktA= new float[csPre.length];
        for(int i=0; i< csPre.length; i++) {
            hadamardProduktA[i]= forgetGate[i]* csPre[i];
        }//PRE: 0.4274353
        //Hadamard-Produkt {inputGate, candidateGate}
        float[] hadamardProduktB= new float[csPre.length];
        for(int i=0; i< inputGate.length; i++) {
            hadamardProduktB[i]= inputGate[i]* candidateGate[i];
        }//PRE: 0.7343723
        //Elementweise Addition beider Hadamard-Produkt
        float[] result= new float[csPre.length];
        for(int i=0; i< inputGate.length; i++) {
            result[i]= hadamardProduktA[i]+ hadamardProduktB[i];
        }//PRE: 1.1618077
        return result;
    }

    @Test
    public void tests() {
        /*
            Google: "LSTM: Ein Rechenbeispiel sowohl für Forwarding als auch für  Backpropagation,
                     wobei ich die Verwendung der rekurrenten  Gewichte(Uf,Ui,Uc,Uo) nachvollziehen kann."

         */
        float DELTA= 1e-3f;
        final Func SIGMOID= x-> (float)(1 / (1 + Math.exp(-x[0])));
        final Func SIGMOID_DERIVATIVE= x->(x[0]*(1-x[0]));
        final Func TANH= x-> (float)((Math.exp(x[0])-Math.exp(-x[0])) / (Math.exp(x[0])+Math.exp(-x[0])));
        final Func TANH_DERIVATIVE= x->(1.0 - (Math.pow(x[0], 2)));
        float x_t= 0.5f;
        float hs_t= 0.0f;
        float cs_t= 0.0f;
        float hs_pre= 0.1f;
        float cs_pre= 0.1f;
        float w_alle= 0.4f;
        float u_f= 0.1f;
        float u_i= 0.2f;
        float u_c= 0.3f;
        float u_o= 0.4f;
        float b_alle= 0.1f;
        int t=0;
        //Forwarding
        Gate forgetGate= new Gate(SIGMOID, new float[]{x_t}, new float[][]{{w_alle}}, new float[]{b_alle}, u_f, new float[]{hs_pre}, t);
        assertEquals(0.577, forgetGate.getResult(t)[0], DELTA);
        Gate inputGate= new Gate(SIGMOID, new float[]{x_t}, new float[][]{{w_alle}}, new float[]{b_alle}, u_i, new float[]{hs_pre}, t);
        assertEquals(0.579, inputGate.getResult(t)[0], DELTA);
        Gate candidateGate= new Gate(TANH, new float[]{x_t}, new float[][]{{w_alle}}, new float[]{b_alle}, u_c, new float[]{hs_pre}, t);
        assertEquals(0.318, candidateGate.getResult(t)[0], DELTA);
        cs_t= forgetGate.getResult(t)[0]* cs_pre+ inputGate.getResult(t)[0]* candidateGate.getResult(t)[0];
        assertEquals(0.242, cs_t, DELTA);
        Gate outputGate= new Gate(SIGMOID, new float[]{x_t}, new float[][]{{w_alle}}, new float[]{b_alle}, u_o, new float[]{hs_pre}, t);
        assertEquals(0.584, outputGate.getResult(t)[0], DELTA);
        hs_t= outputGate.getResult(t)[0] * (float)TANH.calc(cs_t);
        assertEquals(0.138, hs_t, DELTA);

        //Backpropagation u_o
        float del_L_hst= 1.0f;     //Fehlergradient angenommen (vereinfacht für Bsp.)
        float del_hst_ot= (float)TANH.calc(cs_t);
        assertEquals(0.237, del_hst_ot, DELTA);
        float del_ot_zo= (float)SIGMOID_DERIVATIVE.calc(outputGate.getResult(t)[0]);
        assertEquals(0.243, del_ot_zo, DELTA);
        float del_zo_uo= hs_pre;
        assertEquals(0.1, del_zo_uo, DELTA);
        float del_L_uo= del_L_hst* del_hst_ot* del_ot_zo* del_zo_uo;
        assertEquals(0.0057, del_L_uo, DELTA);

        //Backpropagation u_f
        float del_hst_cst= outputGate.getResult(t)[0]* (float)(1- Math.pow(TANH.calc(cs_t), 2));
        assertEquals(0.551, del_hst_cst, DELTA);
        float del_cst_ft= cs_pre;
        assertEquals(0.1, del_cst_ft, DELTA);
        float del_ft_zf= (float)SIGMOID_DERIVATIVE.calc(forgetGate.getResult(t)[0]);
        assertEquals(0.244, del_ft_zf, DELTA);
        float del_zf_uf= hs_pre;
        assertEquals(0.1, del_zf_uf, DELTA);
        float del_L_uf= del_L_hst* del_hst_cst* del_cst_ft* del_ft_zf* del_zf_uf;
        assertEquals(0.00134, del_L_uf, DELTA);

        //Backpropagation u_i
        float del_cst_it= candidateGate.getResult(t)[0];
        assertEquals(0.318, del_cst_it, DELTA);
        float del_it_zt= (float)SIGMOID_DERIVATIVE.calc(inputGate.getResult(t)[0]);
        assertEquals(0.244, del_it_zt, DELTA);
        float del_zi_ui= hs_pre;
        assertEquals(0.1, del_zi_ui, DELTA);
        float del_L_ui= del_L_hst* del_hst_cst* candidateGate.getResult(t)[0]* del_it_zt* del_zi_ui;
        assertEquals(0.00427, del_L_ui, DELTA);

        //Backpropagation u_c
        float del_cst_ct= inputGate.getResult(t)[0];
        assertEquals(0.579, del_cst_ct, DELTA);
        float del_ct_zc= (1- (float)Math.pow(candidateGate.getResult(t)[0], 2));
        assertEquals(0.899, del_ct_zc, DELTA);  // TODO: 24.04.2026 TANH_DERIVATIVE.calc(candidateGate.getResult(t)[0]) kommt nicht zu diesem Ergebnis
        float del_zc_uc= hs_pre;
        assertEquals(0.1, del_zc_uc, DELTA);
        float del_L_uc= del_L_hst* del_hst_cst* del_cst_ct* del_ct_zc* del_zc_uc;
        assertEquals(0.02868, del_L_uc, DELTA);

    }

    @Test
    public void testsCalculationOfRecurrentWeightsAlongForwardingBackpropagation() {
        float DELTA= 1e-3f;
        final Func SIGMOID= x-> (float)(1 / (1 + Math.exp(-x[0])));
        final Func SIGMOID_DERIVATIVE= x->(x[0]*(1-x[0]));
        final Func TANH= x-> (float)((Math.exp(x[0])-Math.exp(-x[0])) / (Math.exp(x[0])+Math.exp(-x[0])));
        final Func TANH_DERIVATIVE= x->(1.0 - (Math.pow(x[0], 2)));
        float x_t= 0.5f;
        float hs_t= 0.0f;
        float cs_t= 0.0f;
        float hs_pre= 0.1f;
        float cs_pre= 0.1f;
        float w_alle= 0.4f;
        float u_f= 0.1f;
        float u_i= 0.2f;
        float u_c= 0.3f;
        float u_o= 0.4f;
        float b_alle= 0.1f;
        int t=0;
        //Forwarding
        Gate forgetGate= new Gate(SIGMOID, new float[]{x_t}, new float[][]{{w_alle}}, new float[]{b_alle}, u_f, new float[]{hs_pre}, t);
        assertEquals(0.577, forgetGate.getResult(t)[0], DELTA);
        Gate inputGate= new Gate(SIGMOID, new float[]{x_t}, new float[][]{{w_alle}}, new float[]{b_alle}, u_i, new float[]{hs_pre}, t);
        assertEquals(0.579, inputGate.getResult(t)[0], DELTA);
        Gate candidateGate= new Gate(TANH, new float[]{x_t}, new float[][]{{w_alle}}, new float[]{b_alle}, u_c, new float[]{hs_pre}, t);
        assertEquals(0.318, candidateGate.getResult(t)[0], DELTA);
        cs_t= forgetGate.getResult(t)[0]* cs_pre+ inputGate.getResult(t)[0]* candidateGate.getResult(t)[0];
        assertEquals(0.242, cs_t, DELTA);
        Gate outputGate= new Gate(SIGMOID, new float[]{x_t}, new float[][]{{w_alle}}, new float[]{b_alle}, u_o, new float[]{hs_pre}, t);
        assertEquals(0.584, outputGate.getResult(t)[0], DELTA);
        hs_t= outputGate.getResult(t)[0] * (float)TANH.calc(cs_t);
        assertEquals(0.138, hs_t, DELTA);

        //Backpropagation u_o
        //Fehlergradient(del_hs_t):
        float del_L_hst= 1.0f;     //Fehlergradient angenommen (vereinfacht für Bsp.)
        //Einfluss des Gates auf hs_t (del_hst_ot):
        float del_hst_ot= (float)TANH.calc(cs_t);
        assertEquals(0.237, del_hst_ot, DELTA);
        //Sigmoid-Ableitung (del_ot_zo):
        float del_ot_zo= (float)SIGMOID_DERIVATIVE.calc(outputGate.getResult(t)[0]);
        assertEquals(0.243, del_ot_zo, DELTA);
        //Einfluss von u_o (del_zo_uo):
        float del_zo_uo= hs_pre;
        assertEquals(0.1, del_zo_uo, DELTA);
        //Gesamtgradient für u_o (del_L_uo):
        float del_L_uo= del_L_hst* del_hst_ot* del_ot_zo* del_zo_uo;
        assertEquals(0.0057, del_L_uo, DELTA);

        //Backpropagation u_f
        //Schritt A: von h_t zu C_t
        float del_hst_cst= outputGate.getResult(t)[0]* (float)(1- Math.pow(TANH.calc(cs_t), 2));
        assertEquals(0.551, del_hst_cst, DELTA);
        //Schritt B: von C_t zu f_t
        float del_cst_ft= cs_pre;
        assertEquals(0.1, del_cst_ft, DELTA);
        //Schritt C: von f_t zum Argument z_f
        float del_ft_zf= (float)SIGMOID_DERIVATIVE.calc(forgetGate.getResult(t)[0]);
        assertEquals(0.244, del_ft_zf, DELTA);
        //Schritt D: von z_f zum rekurrenten Gewicht u_f
        float del_zf_uf= hs_pre;
        assertEquals(0.1, del_zf_uf, DELTA);
        //Das Ergebnis <del_L_uf>
        float del_L_uf= del_L_hst* del_hst_cst* del_cst_ft* del_ft_zf* del_zf_uf;
        assertEquals(0.00134, del_L_uf, DELTA);

        //Backpropagation u_i
        //Einfluss von i_t auf C_t:
        float del_cst_it= candidateGate.getResult(t)[0];
        assertEquals(0.318, del_cst_it, DELTA);
        //Sigmoid-Ableitung (i_t):
        float del_it_zt= (float)SIGMOID_DERIVATIVE.calc(inputGate.getResult(t)[0]);
        assertEquals(0.244, del_it_zt, DELTA);
        //Ableitung nach u_i:
        float del_zi_ui= hs_pre;
        assertEquals(0.1, del_zi_ui, DELTA);
        //Gesamtgradient u_i:
        float del_L_ui= del_L_hst* del_hst_cst* candidateGate.getResult(t)[0]* del_it_zt* del_zi_ui;
        assertEquals(0.00427, del_L_ui, DELTA);

        //Backpropagation u_c
        //Einfluss von ~C_t auf C_t:
        float del_cst_ct= inputGate.getResult(t)[0];
        assertEquals(0.579, del_cst_ct, DELTA);
        //tanh-Ableitung (~C_t):
        float del_ct_zc= (1- (float)Math.pow(candidateGate.getResult(t)[0], 2));
        assertEquals(0.899, del_ct_zc, DELTA);  // TODO: 24.04.2026 TANH_DERIVATIVE.calc(candidateGate.getResult(t)[0]) kommt nicht zu diesem Ergebnis
        //Ableitung nach u_c:
        float del_zc_uc= hs_pre;
        assertEquals(0.1, del_zc_uc, DELTA);
        //Gesamtgradient u_c:
        float del_L_uc= del_L_hst* del_hst_cst* del_cst_ct* del_ct_zc* del_zc_uc;
        assertEquals(0.02868, del_L_uc, DELTA);
    }

    //@Test
    public void testsUWeights() {
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.5f, 0.5f}};
        float[][]wI= new float[][]{{0.1f, 0.1f}};
        float[][]wC= new float[][]{{0.1f, 0.1f}};
        float[][]wO= new float[][]{{0.2f, 0.2f}};
        float u= 0.1f;
        int t= 0;   //the timestep
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
        node.forward(t);
        float aktuellerFehlerAmHS_t= 0.5f;  //delHt
        final Func SIGMOID= x-> (float)(1 / (1 + Math.exp(-x[0])));
        final Func SIGMOID_DERIVATIVE= x->((x[0])*(1-x[0]));
        final Func TANH= x-> (float)((Math.exp(x[0])-Math.exp(-x[0])) / (Math.exp(x[0])+Math.exp(-x[0])));
        //1. Fehler am Output Gate (del_o) berechnen
        float[] outputGatesDerivedMathMulResult= new float[node.outputGate.getZ().length];
        for(int i=0; i<outputGatesDerivedMathMulResult.length; i++) {
            outputGatesDerivedMathMulResult[i]= (float)SIGMOID_DERIVATIVE.calc(node.outputGate.getResult(t)[i]);
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
        float delC_t= delH_t* node.outputGate.getResult(t)[0]* (float)(1- Math.pow(TANH.calc(node.cellState[0]), 2));    //TODO delH_t evt als Vektor/Array
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
            inputGatesDerivedMathMulResult[i]= (float)SIGMOID_DERIVATIVE.calc(node.inputGate.getResult(t)[i]);
        }
        //Schritt B: Fehler am Gate (delI_t)
        float delI_t= delC_t* node.candidateGate.getResult(t)[0]* (float)SIGMOID_DERIVATIVE.calc(SIGMOID.calc(node.inputGate.getZ()[0]));
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
            candidateGatesDerivedMathMulResult[i]= (float)(1- Math.pow(node.candidateGate.getResult(t)[0], 2));
        }
        assertEquals(0.9778, candidateGatesDerivedMathMulResult[0], DELTA);
        //Schritt B: Fehler am Gate (delC_t)
        float FehlerDelC_t= delC_t* node.inputGate.getResult(t)[0]* (float)(1- Math.pow(node.candidateGate.getResult(t)[0], 2));
        assertEquals(0.1496, FehlerDelC_t, DELTA);

        //6. Gradient und Update für U_c
        //Schritt C: Der Gradient (delU_c)
        float delU_c= FehlerDelC_t* node.prevCell.hiddenState[0];     //TODO delU_c evt als Vektor/Array
        assertEquals(0.01496, delU_c, DELTA);
        //Schritt D: Gewicht anpassen
        float U_c_neu= U_c - (LERNRATE* delU_c);
        assertEquals(0.098504, U_c_neu, DELTA);

        //Vervollständigte Berechnung für rekurrentes Gewicht am Output Gate
        //Schritt A: Ableitung der Aktivierungsfunktion (sigmoid')
        for(int i=0; i<candidateGatesDerivedMathMulResult.length; i++) {
            candidateGatesDerivedMathMulResult[i]= node.outputGate.getResult(t)[0]* (1- node.outputGate.getResult(t)[0]);
        }
        assertEquals(0.2445, candidateGatesDerivedMathMulResult[0], DELTA);

        //Schritt B: Fehler am Gate (delO_t)
        float FehlerDelO_t= delH_t* (float)TANH.calc(node.cellState[0])* (float)(SIGMOID_DERIVATIVE.calc(node.outputGate.getResult(t)[0]));
        assertEquals(0.00976, FehlerDelO_t, DELTA);

        //Gradient und Update für U_o
        //Schritt C: Der Gradient (delU_o)
        delU_o= FehlerDelO_t* node.prevCell.hiddenState[0];     //TODO delU_c evt als Vektor/Array
        assertEquals(0.000976, delU_o, DELTA);
        //Schritt D: Gewicht anpassen
        U_o_neu= U_o - (LERNRATE* delU_o);
        assertEquals(0.0999024, U_o_neu, DELTA);
        System.out.println("testsUWeights() says:");
        System.out.println("\tParameter\t\tAlter Wert\tÄnderung\t\tNeuer Wert");
        System.out.println("\tUf(Forget)\t\t"+U_f+"\t\t\t"+delU_f+"\t"+U_f_neu);
        System.out.println("\tUi(Input)\t\t"+U_i+"\t\t\t"+delU_i+"\t"+U_i_neu);
        System.out.println("\tUc(Candidate)\t"+U_c+"\t\t\t"+delU_c+"\t\t"+U_c_neu);
        System.out.println("\tUo(Output)\t\t"+U_o+"\t\t\t"+delU_o+"\t"+U_o_neu);
    }

    //@Test
    public void testsBackpropagationAdjustingBiases() {
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.5f, 0.5f}};
        float[][]wI= new float[][]{{0.1f, 0.1f}};
        float[][]wC= new float[][]{{0.1f, 0.1f}};
        float[][]wO= new float[][]{{0.2f, 0.2f}};
        float u= 0.1f;
        int t= 0;   //the timestep
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
        assertEquals(0.6792f, node.forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.5374f, node.inputGate.getResult(t)[0], DELTA);
        assertEquals(0.5744f, node.outputGate.getResult(t)[0], DELTA);
        //2. Der neue Zellzustand-Kandidat (~C[t])
        assertEquals(0.149f, node.candidateGate.getResult(t)[0], DELTA);
        node.forward(t);
        //3. Update des Cell State (C[t])
        assertEquals(0.080f, node.cellState[0], DELTA);
        //4. Output / Hidden State (h[t])
        assertEquals(0.046f, node.hiddenState[0], DELTA);
        //Der Fehler am Output (del_h[t])
        float y= 0.5f;
        node.backpropagateGamma(node.xt, wF, wI, wC, wO,  new float[]{0f}, y, 2, t);
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

    //@Test
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
        int t= 0;
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
        node.forward(t);
        /* DONE: 17.04.2026 Ich befürchte ich sollte die Backpropagation nochmal komplett neu aufarbeiten
            Zuerst mit einem nachvollziehbaren Bsp. von Google
            "lstm gib mir ein Rechenbeispiel mit Zahlenbeispiel für eine lstm-row mit wF={0.5, 0.5}, wI={0.1, 0.1}, wC={0.1, 0.1}, wO={0.2, 0.2}, bF={0.0, 0.0}, bI={0.0, 0.0}, bC={0.0, 0.0}, bO={0.0, 0.0} und U=0.1 und input={1,0.5}. Verwende bei den Gates auch die Aktivierungsfunktionen"
         */
        node.backpropagateBeta(node.xt, wF, wI, wC, wO,  new float[]{0f}, 1f, 1, t);

        assertArrayEquals(new float[]{0.4876f, 0.4938f}, wF[0], DELTA);
        assertArrayEquals(new float[]{0.0979f, 0.0989f}, wI[0], DELTA);
        assertArrayEquals(new float[]{0.07f, 0.085f}, wC[0], DELTA);
        assertArrayEquals(new float[]{0.1981f, 0.1990f}, wO[0], DELTA);
    }

    //@Test
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
        int t=0;
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
        node.forward(t);
        /* DONE: 17.04.2026 Ich befürchte ich sollte die Backpropagation nochmal komplett neu aufarbeiten
            Zuerst mit einem nachvollziehbaren Bsp. von Google
            "lstm gib mir ein Rechenbeispiel mit Zahlenbeispiel für eine lstm-row mit wF={0.5, 0.5}, wI={0.1, 0.1}, wC={0.1, 0.1}, wO={0.2, 0.2}, bF={0.0, 0.0}, bI={0.0, 0.0}, bC={0.0, 0.0}, bO={0.0, 0.0} und U=0.1 und input={1,0.5}. Verwende bei den Gates auch die Aktivierungsfunktionen"
         */
        node.backpropagateBeta(node.xt, wF, wI, wC, wO, new float[]{0f}, 1f, -1, t);

        assertArrayEquals(new float[]{0.4876f, 0.4938f}, wF[0], DELTA);
        assertArrayEquals(new float[]{0.0979f, 0.0989f}, wI[0], DELTA);
        assertArrayEquals(new float[]{0.07f, 0.085f}, wC[0], DELTA);
        assertArrayEquals(new float[]{0.1981f, 0.1990f}, wO[0], DELTA);
    }

    //@Test
    public void testsForwardingA() {
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.5f, 0.5f}};
        float[][]wI= new float[][]{{0.1f, 0.1f}};
        float[][]wC= new float[][]{{0.1f, 0.1f}};
        float[][]wO= new float[][]{{0.2f, 0.2f}};
        int t= 0;   //the timestep
        LSTM_cell node= new LSTM_cell(
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                new float[]{1f,0.5f}, new float[]{0,1,0}, wF, wI, wC, wO,
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                new float[]{0.f, 0.f}, new float[]{0.f, 0.f}, 0.1f, false, false, false);
        node.forward(t);
        assertEquals(0.6792f, node.forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.5374f, node.inputGate.getResult(t)[0], DELTA);
        assertEquals(0.1489f, node.candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.5744f, node.outputGate.getResult(t)[0], DELTA);
        assertArrayEquals(new float[]{0.08f, 0.08f}, node.cellState, DELTA);
        assertArrayEquals(new float[]{0.0459f, 0.0459f}, node.hiddenState, DELTA);
    }
    //@Test
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
        int t= 0;   //the timestep
        LSTM_cell node= new LSTM_cell(
                new float[]{0.0f},      //Um Rechnung überschaubar zu halten, setzen wir die Größe des
                new float[]{0.0f},      //  Hidden-/Cell State (Anzahl der Neuronen in der LSTM-Zelle) auf 1.
                new float[]{0.5f,0.3f},   //(Ein 2D-Vektor)
                new float[]{0,1,0},
                wF, wI, wC, wO,
                new float[]{0.5f}, new float[]{0.f},
                new float[]{0.f}, new float[]{0.1f},
                u, false, false, true);
        node.forward(t);
        /*DONE Fehler wurde mit [FLAG]<mulMitGetrimmtemInput> in Gate().calculateResult(float,float[],boolean) korrigiert
            IST: node.forgetGate.getResult(t)[0] kommt raus [0.6456563]
            SOLL: node.forgetGate.getResult(t)[0] muss ca [0.648] rauskommen
            Bsp-Rechnung:       0.1*0.0 + 0.1*0.5 + 0.2*0.3 + 0.5= 0.61
            Funktion-Rechnung:  0.1*0.0 + 0.2*0.5 +
                                0.1*0.0 +
                                0.5
                                =0.6
         */
        assertEquals(0.6479f, node.forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.6659f, node.inputGate.getResult(t)[0], DELTA);
        assertEquals(0.3799f, node.candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.5449f, node.outputGate.getResult(t)[0], DELTA);
        assertArrayEquals(new float[]{0.253f}, node.cellState, DELTA);
        assertArrayEquals(new float[]{0.135f}, node.hiddenState, DELTA);
    }

    //@Test
    public void testsForwardingC() {
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.1f, 0.2f}};
        float[][]wI= new float[][]{{0.9f, 0.8f}};
        float[][]wC= new float[][]{{0.5f, 0.5f}};
        float[][]wO= new float[][]{{0.1f, 0.1f}};
        float u= 0.1f;
        int t= 0;   //the timestep
        LSTM_cell node= new LSTM_cell(
                new float[]{0.0f},      //Um Rechnung überschaubar zu halten, setzen wir die Größe des
                new float[]{0.0f},      //  Hidden-/Cell State (Anzahl der Neuronen in der LSTM-Zelle) auf 1.
                new float[]{0.5f,0.3f},   //(Ein 2D-Vektor)
                new float[]{0,1,0},
                wF, wI, wC, wO,
                new float[]{0.5f}, new float[]{0.f},
                new float[]{0.f}, new float[]{0.1f},
                u, false, false, false);
        node.forward(t);
        assertEquals(0.648f, node.forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.666f, node.inputGate.getResult(t)[0], DELTA);
        assertEquals(0.380f, node.candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.545f, node.outputGate.getResult(t)[0], DELTA);
        assertArrayEquals(new float[]{0.253f}, node.cellState, DELTA);
        assertArrayEquals(new float[]{0.135f}, node.hiddenState, DELTA);
    }

    //@Test
    public void testsForwardingD() {
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.2f, 0.3f, 0.1f}};
        float[][]wI= new float[][]{{0.1f, -0.1f, 0.2f}};
        float[][]wC= new float[][]{{-0.3f, 0.1f, 0.5f}};
        float[][]wO= new float[][]{{0.4f, 0.2f, -0.1f}};
        float u= 0.1f;
        int t= 0;   //the timestep
        LSTM_cell node= new LSTM_cell(
                new float[]{0.f, 0.f, 0.f},      //Um Rechnung überschaubar zu halten, setzen wir die Größe des
                new float[]{0.f, 0.f, 0.f},      //  Hidden-/Cell State (Anzahl der Neuronen in der LSTM-Zelle) auf 1.
                new float[]{0.5f,-0.2f, 0.1f},   //(Ein 3D-Vektor)
                new float[]{0,1,0},
                wF, wI, wC, wO,
                new float[]{0.1f}, new float[]{-0.1f},
                new float[]{0.f}, new float[]{0.2f},
                u, false, false, false);
        node.forward(t);
        assertEquals(0.537f, node.forgetGate.getResult(t)[0], DELTA);
        assertEquals(0.497f, node.inputGate.getResult(t)[0], DELTA);
        assertEquals(-0.119f, node.candidateGate.getResult(t)[0], DELTA);
        assertEquals(0.587f, node.outputGate.getResult(t)[0], DELTA);
        assertEquals(-0.059f, node.cellState[0], DELTA);       /* DONE: 16.04.2026 wieso enthält node.cellState[] mehr als einen Wert???
         *       evt weil preCS 3 dimensional ist*/
        assertEquals(-0.0346f, node.hiddenState[0], DELTA);    /* DONE: 16.04.2026 wieso enthält node.hiddenState[] mehr als einen Wert???
         *       evt weil preHS 3 dimensional ist*/
    }

    //@Test
    public void testsForwardingE() {
        float DELTA= 1e-3f;
        float[][]wF= new float[][]{{0.1f, -0.2f, 0.5f}};
        float[][]wI= new float[][]{{0.2f, 0.3f, -0.1f}};
        float[][]wC= new float[][]{{0.8f, -0.5f, 0.0f}};
        float[][]wO= new float[][]{{0.4f, 0.1f, 0.2f}};
        float u= 0.1f;
        int t= 0;   //the timestep
        LSTM_cell node= new LSTM_cell(
                new float[]{0.2f},      //Um Rechnung überschaubar zu halten, setzen wir die Größe des
                new float[]{0.1f},      //  Hidden-/Cell State (Anzahl der Neuronen in der LSTM-Zelle) auf 1.
                new float[]{0.5f,-0.2f},   //(Ein 2D-Vektor)
                new float[]{0,1,0},
                wF, wI, wC, wO,
                new float[]{0.1f}, new float[]{0.2f},
                new float[]{0.f}, new float[]{-0.1f},
                u, true, false, true);
        node.forward(t);
        assertEquals(0.562f, node.forgetGate.getResult(t)[0], DELTA);        // DONE: 24.04.2026 die Gates wurden umgerüstet, dass sie statt der summe über die preHS, das Produkt aus entspr. <u> und dem preHS zu result addiert => entspr ändert sich auch leicht der Zielwert des Resultats
        assertEquals(0.559f, node.inputGate.getResult(t)[0], DELTA);         // DONE: 24.04.2026 die Gates wurden umgerüstet, dass sie statt der summe über die preHS, das Produkt aus entspr. <u> und dem preHS zu result addiert => entspr ändert sich auch leicht der Zielwert des Resultats
        assertEquals(0.469f, node.candidateGate.getResult(t)[0], DELTA);     // DONE: 24.04.2026 die Gates wurden umgerüstet, dass sie statt der summe über die preHS, das Produkt aus entspr. <u> und dem preHS zu result addiert => entspr ändert sich auch leicht der Zielwert des Resultats
        assertEquals(0.527f, node.outputGate.getResult(t)[0], DELTA);        // DONE: 24.04.2026 die Gates wurden umgerüstet, dass sie statt der summe über die preHS, das Produkt aus entspr. <u> und dem preHS zu result addiert => entspr ändert sich auch leicht der Zielwert des Resultats
        assertEquals(0.375f, node.cellState[0], DELTA);       /* DONE: 16.04.2026 wieso enthält node.cellState[] mehr als einen Wert???
         *                  evt weil preCS 3 dimensional ist*/
//        assertArrayEquals(new float[]{-0.0181f}, node.cellState, DELTA);
        assertEquals(0.189f, node.hiddenState[0], DELTA);    /* DONE: 16.04.2026 wieso enthält node.hiddenState[] mehr als einen Wert???
         *                  evt weil preHS 3 dimensional ist*/
    }
}
