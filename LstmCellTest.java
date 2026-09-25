package lstm2609251410;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class LstmCellTest {

    @Test
    void testsLstmMitJointLayerNormalizationVollstaendigesBptt() {
        float[]h0= new float[]{0f, 0f};
        float[]c0= new float[]{0f, 0f};

        float[][] wf = new float[][]{{0.4f, 0.1f, 0.0f}, {-0.3f, 0.2f,  0.1f}};
        float[][] uf = new float[][]{{0.2f, -0.1f}, {0.0f, 0.4f}};
        float[][] wi = new float[][]{{0.2f, -0.1f, 0.3f}, {0.1f, 0.4f,  -0.2f}};
        float[][] ui = new float[][]{{0.1f, 0.2f}, {-0.1f, 0.3f}};
        float[][] wc = new float[][]{{-0.2f, 0.3f, 0.1f}, {0.1f, -0.1f,  0.5f}};
        float[][] uc = new float[][]{{0.3f, 0.0f}, {-0.2f, 0.1f}};
        float[][] wo = new float[][]{{0.1f, -0.2f, 0.4f}, {0.3f, 0.1f,  -0.1f}};
        float[][] uo = new float[][]{{-0.1f, 0.2f}, {0.1f, 0.3f}};
        float[] xt1= new float[]{1f, 0f, -1f};

        LstmCell cell= new LstmCell(2, wf, wi, wc, wo, uf, ui, uc, uo);
        cell.hs_prev= new float[]{0.f, 0.f};
        cell.cs_prev= new float[]{0.f, 0.f};
        int TIME_STEP= 0;
        int TIME_STEPS= 1;
        cell.forwardPass(TIME_STEP, xt1);
        //Schritt A: Lineare Kombination (Pre-Activation)   getestet von testsLstmMitJointLayerNormalizationVerkuerzt()
        //Schritt B: Joint Layer Normalization              getestet von testsLstmMitJointLayerNormalizationVerkuerzt()
        //Schritt C: Splitten und Aktivierungsfunktionen    getestet von testsLstmMitJointLayerNormalizationVerkuerzt()
        //Schritt D: Update der States (c1 und h1)          getestet von testsLstmMitJointLayerNormalizationVerkuerzt()

        float[] target= new float[]{0f, 0f};
        // DONE: 22.09.2026 Natürlich resultiert hier ein Fehler-Vektor: Fehler pro Feature-Kombination
        LossFunc MSE= (hs, y) -> {
            float[] res= new float[hs.length];
            for(int i=0; i< res.length; i++) {
                res[i]= hs[i]-y[i];
                res[i]*= 2;
            }
            return res;
        };
        float[][]bpttRes= cell.backwardPass(TIME_STEP, MSE, target, xt1, cell.fActivated[TIME_STEP], cell.iActivated[TIME_STEP], cell.cActivated[TIME_STEP], cell.oActivated[TIME_STEP]);
        Adam adam= new Adam(TIME_STEPS, cell.delL_delWf[TIME_STEP], wf);
        float[][]optimizedWeights= adam.calc(TIME_STEP+1, cell.delL_delWf[TIME_STEP], wf);
        Assertions.assertEquals(0.401f, optimizedWeights[0][0], 1e-04);
        Assertions.assertEquals(0.1f, optimizedWeights[0][1], 1e-04);
        Assertions.assertEquals(-0.001f, optimizedWeights[0][2], 1e-04);
        Assertions.assertEquals(-0.301f, optimizedWeights[1][0], 1e-04);
        Assertions.assertEquals(0.2f, optimizedWeights[1][1], 1e-04);
        Assertions.assertEquals(0.101f, optimizedWeights[1][2], 1e-04);
        /* TODO: 25.09.2026
            1.) Bei längeren Sequenzen (t>1) : Die Schleife läuft weiter
                Wenn deine Sequenz zum Beispiel aus 5 Zeitschriften besteht,
                bist du jetzt erst am Ende von Schritt 1 angekommen:
                    Rekursive Weiterführung:
                        Du nimmst die eben berechneten Gradienten delL/delHS(t:0) und delL/delCS(t:0) und
                        steckst sie als neuen "Start-Fehler" in den nächsten Zeitschritt weiter links (t:0).
                    Gradienten-Akkumulation:
                        Da die Gewichte/-Matrizen W* und U* über die gesamte Sequenz hinweg geteilt/geshared werden,
                        musst du die Gradienten (grad_W und grad_U) aus allen Zeitschriften aufsummieren:
                            W_total= delL_delWf(t)+ delL_delWi(t)+ delL_delWc(t)+ delL_delWo(t)
                            U_total= delL_delUf(t)+ delL_delUi(t)+ delL_delUc(t)+ delL_delUo(t)
            2.) Nach dem vollständigen BPTT: Der Optimierungs-Schritt
                Sobald du den Fehler durch die gesamte Sequenz (alle Zeitschritte) geworfen und
                die finalen Gradienten zusammengefasst hast, kommt der eigentliche Lernschritt des Netzes:
                    Gewichtsaktualisierung (Weight Update):
                        Ein Optimizer (wie Adam oder Stochastic Gradient Descent) nutzt die berechneten Gradienten,
                        um die Gewichtsmatrizen W und U anzupassen:
                            W_neu = W_alt - η * W_total
            *
            Zusammenfassung:
                Für diesen einen Zeitschritt:
                    Bist du fertig!
                    Die Gradienten für W und U liegen vor.
                Für das gesamte Training:
                    Jetzt würden die Gradienten an den Optimizer übergeben werden und das Netz beginnt im nächsten
                    Epoch wieder von vorn mit dem nächsten Forward-Pass.
         */
    }

    @Test
    void testsLstmMitJointLayerNormalizationVerkuerzt() {
        float[]h0= new float[]{0f, 0f};
        float[]c0= new float[]{0f, 0f};

        float[][] wf = new float[][]{{0.4f, 0.1f, 0.0f}, {-0.3f, 0.2f,  0.1f}};
        float[][] uf = new float[][]{{0.2f, -0.1f}, {0.0f, 0.4f}};
        float[][] wi = new float[][]{{0.2f, -0.1f, 0.3f}, {0.1f, 0.4f,  -0.2f}};
        float[][] ui = new float[][]{{0.1f, 0.2f}, {-0.1f, 0.3f}};
        float[][] wc = new float[][]{{-0.2f, 0.3f, 0.1f}, {0.1f, -0.1f,  0.5f}};
        float[][] uc = new float[][]{{0.3f, 0.0f}, {-0.2f, 0.1f}};
        float[][] wo = new float[][]{{0.1f, -0.2f, 0.4f}, {0.3f, 0.1f,  -0.1f}};
        float[][] uo = new float[][]{{-0.1f, 0.2f}, {0.1f, 0.3f}};
        float[] xt1= new float[]{1f, 0f, -1f};

        LstmCell cell= new LstmCell(2, wf, wi, wc, wo, uf, ui, uc, uo);
        cell.hs_prev= new float[]{0.f, 0.f};
        cell.cs_prev= new float[]{0.f, 0.f};
        int TIME_STEP= 0;
        cell.forwardPass(TIME_STEP, xt1);

        //Schritt A: Lineare Kombination (Pre-Activation)
        Assertions.assertEquals(0.4f, cell.fNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.4f, cell.fNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.1f, cell.iNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.3f, cell.iNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.3f, cell.cNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.4f, cell.cNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.3f, cell.oNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.4f, cell.oNetto[TIME_STEP][1], 1e-02);

        //Schritt B: Joint Layer Normalization
        Assertions.assertEquals(-0.1491f, cell.iScaledAndMoved[TIME_STEP][0], 1e-04);
        Assertions.assertEquals(1.0435f, cell.iScaledAndMoved[TIME_STEP][1], 1e-04);
        Assertions.assertEquals(1.3416f, cell.fScaledAndMoved[TIME_STEP][0], 1e-04);
        Assertions.assertEquals(-1.0435f, cell.fScaledAndMoved[TIME_STEP][1], 1e-04);
        Assertions.assertEquals(-0.7454f, cell.cScaledAndMoved[TIME_STEP][0], 1e-04);
        Assertions.assertEquals(-1.0435f, cell.cScaledAndMoved[TIME_STEP][1], 1e-04);
        Assertions.assertEquals(-0.7454f, cell.oScaledAndMoved[TIME_STEP][0], 1e-04);
        Assertions.assertEquals(1.3416f, cell.oScaledAndMoved[TIME_STEP][1], 1e-04);

        //Schritt C: Splitten und Aktivierungsfunktionen
        Assertions.assertEquals(0.7927f, cell.fActivated[TIME_STEP][0], 1e-04);
        Assertions.assertEquals(0.2606f, cell.fActivated[TIME_STEP][1], 1e-03);
        Assertions.assertEquals(0.4627f, cell.iActivated[TIME_STEP][0], 1e-03);
        Assertions.assertEquals(0.7394f, cell.iActivated[TIME_STEP][1], 1e-03);
        Assertions.assertEquals(-0.6324f, cell.cActivated[TIME_STEP][0], 1e-04);
        Assertions.assertEquals(-0.7797f, cell.cActivated[TIME_STEP][1], 1e-03);
        Assertions.assertEquals(0.322f, cell.oActivated[TIME_STEP][0], 1e-03);
        Assertions.assertEquals(0.7927f, cell.oActivated[TIME_STEP][1], 1e-04);

        //Schritt D: Update der States (c1 und h1)
        Assertions.assertEquals(-0.2926f, cell.cs_t[TIME_STEP][0], 1e-04);
        Assertions.assertEquals(-0.5765f, cell.cs_t[TIME_STEP][1], 1e-03);
        Assertions.assertEquals(-0.0917f, cell.hs_t[TIME_STEP][0], 1e-03);
        Assertions.assertEquals(-0.4124f, cell.hs_t[TIME_STEP][1], 1e-03);

        float[] target= new float[]{0f, 0f};
        // DONE: 22.09.2026 Natürlich resultiert hier ein Fehler-Vektor: Fehler-Vektor pro Feature(-Kombination)
        LossFunc MSE= (hs, y) -> {
            float[] res= new float[hs.length];
            for(int i=0; i< res.length; i++) {
                res[i]= hs[i]-y[i];
                res[i]*= 2;
            }
            return res;
        };
        float[][]bpttRes= cell.backwardPass(TIME_STEP, MSE, target, xt1, cell.fActivated[TIME_STEP], cell.iActivated[TIME_STEP], cell.cActivated[TIME_STEP], cell.oActivated[TIME_STEP]);
        Assertions.assertEquals(0.0264f, bpttRes[0][0], 1e-03);
        Assertions.assertEquals(0.107f, bpttRes[0][1], 1e-03);
        Assertions.assertEquals(-0.043f, bpttRes[1][0], 1e-04);
        Assertions.assertEquals(-0.1241f, bpttRes[1][1], 1e-03);
        Assertions.assertEquals(-0.1603f, cell.delL_delWf[TIME_STEP][0][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delWf[TIME_STEP][0][1], 1e-03);
        Assertions.assertEquals(0.1603f, cell.delL_delWf[TIME_STEP][0][2], 1e-03);
        Assertions.assertEquals(0.1191f, cell.delL_delWf[TIME_STEP][1][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delWf[TIME_STEP][1][1], 1e-03);
        Assertions.assertEquals(-0.1191f, cell.delL_delWf[TIME_STEP][1][2], 1e-03);
        Assertions.assertEquals(0.107f, bpttRes[0][1], 1e-03);
        Assertions.assertEquals(-0.043f, bpttRes[1][0], 1e-04);

        Assertions.assertEquals(0.0397f, cell.delL_delWi[TIME_STEP][0][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delWi[TIME_STEP][0][1], 1e-03);
        Assertions.assertEquals(-0.0397f, cell.delL_delWi[TIME_STEP][0][2], 1e-03);
        Assertions.assertEquals(0.0881f, cell.delL_delWi[TIME_STEP][1][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delWi[TIME_STEP][1][1], 1e-03);
        Assertions.assertEquals(-0.0881f, cell.delL_delWi[TIME_STEP][1][2], 1e-03);

        Assertions.assertEquals(0.0392f, cell.delL_delWc[TIME_STEP][0][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delWc[TIME_STEP][0][1], 1e-03);
        Assertions.assertEquals(-0.0392f, cell.delL_delWc[TIME_STEP][0][2], 1e-03);
        Assertions.assertEquals(-0.2926f, cell.delL_delWc[TIME_STEP][1][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delWc[TIME_STEP][1][1], 1e-03);
        Assertions.assertEquals(0.2926f, cell.delL_delWc[TIME_STEP][1][2], 1e-03);

        Assertions.assertEquals(0.1181f, cell.delL_delWo[TIME_STEP][0][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delWo[TIME_STEP][0][1], 1e-03);
        Assertions.assertEquals(-0.1181f, cell.delL_delWo[TIME_STEP][0][2], 1e-03);
        Assertions.assertEquals(0.0487f, cell.delL_delWo[TIME_STEP][1][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delWo[TIME_STEP][1][1], 1e-03);
        Assertions.assertEquals(-0.0487f, cell.delL_delWo[TIME_STEP][1][2], 1e-03);

        /*IMPORTANT Hinweis: Das ist völlig korrekt für den ersten Zeitschritt ($t=1$), da ein Nullvektor als
           Start-Hidden-State keine historischen Einflüsse über die $U$-Matrizen transportieren kann.
           Erst ab t:2 (wenn HS(t) != 0) liefert grad_U Werte ungleich Null.*/
        Assertions.assertEquals(0.0f, cell.delL_delUf[TIME_STEP][0][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delUf[TIME_STEP][0][1], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delUf[TIME_STEP][1][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delUf[TIME_STEP][1][1], 1e-03);

        Assertions.assertEquals(0.0f, cell.delL_delUi[TIME_STEP][0][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delUi[TIME_STEP][0][1], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delUi[TIME_STEP][1][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delUi[TIME_STEP][1][1], 1e-03);

        Assertions.assertEquals(0.0f, cell.delL_delUc[TIME_STEP][0][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delUc[TIME_STEP][0][1], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delUc[TIME_STEP][1][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delUc[TIME_STEP][1][1], 1e-03);

        Assertions.assertEquals(0.0f, cell.delL_delUo[TIME_STEP][0][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delUo[TIME_STEP][0][1], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delUo[TIME_STEP][1][0], 1e-03);
        Assertions.assertEquals(0.0f, cell.delL_delUo[TIME_STEP][1][1], 1e-03);

        Assertions.assertEquals(0.107f, bpttRes[0][1], 1e-03);
        Assertions.assertEquals(-0.043f, bpttRes[1][0], 1e-04);
    }

    @Test
    void testsLstmMitJointLayerNormalization() {
        // TODO: 21.09.2026 C:\Users\sven-.SVEN\Desktop\GeminiLstmPreActivationNormalisationForwardingBpttLangeErklaerung\LSTM-JointLayerNormalisierungBspRechnung\LSTM mit Layer Normalization Rechenbeispiel - Google Gemini.htm
        float[]h0= new float[]{0f, 0f};
        float[]c0= new float[]{0f, 0f};

        float[][] wf = new float[][]{{0.4f, 0.1f, 0.0f}, {-0.3f, 0.2f,  0.1f}};
        float[][] uf = new float[][]{{0.2f, -0.1f}, {0.0f, 0.4f}};
        float[][] wi = new float[][]{{0.2f, -0.1f, 0.3f}, {0.1f, 0.4f,  -0.2f}};
        float[][] ui = new float[][]{{0.1f, 0.2f}, {-0.1f, 0.3f}};
        float[][] wc = new float[][]{{-0.2f, 0.3f, 0.1f}, {0.1f, -0.1f,  0.5f}};
        float[][] uc = new float[][]{{0.3f, 0.0f}, {-0.2f, 0.1f}};
        float[][] wo = new float[][]{{0.1f, -0.2f, 0.4f}, {0.3f, 0.1f,  -0.1f}};
        float[][] uo = new float[][]{{-0.1f, 0.2f}, {0.1f, 0.3f}};
        float[] xt1= new float[]{1f, 0f, -1f};
        float[] xt2= new float[]{0.5f, 1f, 0f};
        LstmCell cell= new LstmCell(2, wf, wi, wc, wo, uf, ui, uc, uo);
        cell.hs_prev= new float[]{0.f, 0.f};
        cell.cs_prev= new float[]{0.f, 0.f};
        int TIME_STEP= 0;
        cell.forwardPass(TIME_STEP, xt1);

        //Schritt A: Lineare Kombination (Pre-Activation)
        Assertions.assertEquals(0.4f, cell.fNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.4f, cell.fNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.1f, cell.iNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.3f, cell.iNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.3f, cell.cNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.4f, cell.cNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.3f, cell.oNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.4f, cell.oNetto[TIME_STEP][1], 1e-02);

        //Schritt B: Joint Layer Normalization
        Assertions.assertEquals(-0.1491f, cell.iScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(1.0435f, cell.iScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(1.3416f, cell.fScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-1.0435f, cell.fScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.7454f, cell.cScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-1.0435f, cell.cScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.7454f, cell.oScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(1.3416f, cell.oScaledAndMoved[TIME_STEP][1], 1e-02);

        //Schritt C: Splitten und Aktivierungsfunktionen
        Assertions.assertEquals(0.7927f, cell.fActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.2606f, cell.fActivated[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.4627f, cell.iActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.7394f, cell.iActivated[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.6324f, cell.cActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.7797f, cell.cActivated[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.322f, cell.oActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.7927f, cell.oActivated[TIME_STEP][1], 1e-02);

        //Schritt D: Update der States (c1 und h1)
        Assertions.assertEquals(-0.2926f, cell.cs_t[TIME_STEP][0], 1e-03);
        Assertions.assertEquals(-0.5765f, cell.cs_t[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.0917f, cell.hs_t[TIME_STEP][0], 1e-03);
        Assertions.assertEquals(-0.4124f, cell.hs_t[TIME_STEP][1], 1e-02);

        TIME_STEP= 1;
        cell.forwardPass(TIME_STEP, xt2);

        //Schritt A: Lineare Kombination (Pre-Activation)
        Assertions.assertEquals(0.3228f, cell.fNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.115f, cell.fNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.0916f, cell.iNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.3355f, cell.iNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.1725f, cell.cNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.0729f, cell.cNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.2233f, cell.oNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.1171, cell.oNetto[TIME_STEP][1], 1e-02);

        //3. Korrigierter normalisierter Vektor
        Assertions.assertEquals(-0.7453f, cell.iScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(1.4172f, cell.iScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(1.3529f, cell.fScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.8638f, cell.fScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.5919f, cell.cScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.6506f, cell.cScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-1.4122f, cell.oScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.3114f, cell.oScaledAndMoved[TIME_STEP][1], 1e-02);

        //Schritt C: Splitten und Aktivierungsfunktionen
        Assertions.assertEquals(0.7946f, cell.fActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.2965f, cell.fActivated[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.3222f, cell.iActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.8049f, cell.iActivated[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.5305f, cell.cActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.5724f, cell.cActivated[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.1959f, cell.oActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.5772f, cell.oActivated[TIME_STEP][1], 1e-02);

        //Schritt D: Update der States (c1 und h1) nach: Korrigierter normalisierter Vektor
        Assertions.assertEquals(-0.0616f, cell.cs_t[TIME_STEP][0], 1e-03);
        Assertions.assertEquals(-0.6316f, cell.cs_t[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.0121f, cell.hs_t[TIME_STEP][0], 1e-03);
        Assertions.assertEquals(-0.3222f, cell.hs_t[TIME_STEP][1], 1e-02);

        float[] target= new float[]{0f, 0f};
        // DONE: 22.09.2026 Natürlich resultiert hier ein Fehler-Vektor: Fehler pro Feature-Kombination
        LossFunc MSE= (hs, y) -> {
            float[] res= new float[hs.length];
            for(int i=0; i< res.length; i++) {
                res[i]= hs[i]-y[i];
            }
            return res;
        };
        cell.backwardPass(TIME_STEP, MSE, target, xt2, cell.fActivated[TIME_STEP], cell.iActivated[TIME_STEP], cell.cActivated[TIME_STEP], cell.oActivated[TIME_STEP]);
    }

    @Test
    void testsLstmForwardingAndBPTT() {
        /* TODO: 20.09.2026
            Auf Rechenergebnisse testen:
            C:\Users\sven-.SVEN\Desktop\GeminiLstmPreActivationNormalisationForwardingBpttLangeErklaerung\LSTM-RechenbeispielZumTesten
         */
        float[][] wf = new float[][]{{0.2f, 0.1f, -0.1f}, {0.0f, 0.3f,  0.2f}};
        float[][] wi = new float[][]{{0.1f, -0.2f,  0.0f}, {0.4f,  0.1f, -0.1f}};
        float[][] wc = new float[][]{{0.5f,  0.0f, 0.1f}, {0.2f, -0.3f, 0.0f}};
        float[][] wo = new float[][]{{-0.1f, 0.2f,  0.3f}, {0.0f, 0.1f, -0.2f}};
        float[][] uf = new float[][]{{0.1f, 0.0f}, {0.0f, 0.1f}};
        float[][] ui = new float[][]{{0.2f, 0.1f}, {-0.1f, 0.0f}};
        float[][] uc = new float[][]{{0.0f,  0.2f}, {0.1f, -0.1f}};
        float[][] uo = new float[][]{{0.1f, -0.1f}, {0.0f, 0.2f}};
        LstmCell cell= new LstmCell(1, wf, wi, wc, wo, uf, ui, uc, uo);
        float[] xt= new float[]{1f, 0.5f, -1.0f};      //IMPORTANT: Faustregel: x_t beschreibt, was reinkommt (die Input-Features)
        cell.hs_prev= new float[]{0.f, 0.f};
        cell.cs_prev= new float[]{0.f, 0.f};
        int TIME_STEP= 0;
        cell.forwardPass(TIME_STEP, xt);
        //2. Affine Transformationen (Pre-Activations)
        Assertions.assertEquals(0.35f, cell.fNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.05f, cell.fNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.0f, cell.iNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.55f, cell.iNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.4f, cell.cNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.05f, cell.cNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.3f, cell.oNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.25f, cell.oNetto[TIME_STEP][1], 1e-02);

        /* IMPORTANT: 25.09.2026 Ich glaube es liegt daran, dass ich im Gemini-Rechenbsp nicht erwähnt habe,
                                 dass es sich um Joint Layer Normalization hadelt*/
        Assertions.assertEquals(1f, cell.fScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-1f, cell.fScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-1f, cell.iScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(1f, cell.iScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(1f, cell.cScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-1f, cell.cScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.832f, cell.oScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.832f, cell.oScaledAndMoved[TIME_STEP][1], 1e-02);

        //3. Ausführliche Pre-Activation-Normalisierung (LayerNorm pro Gatter)

        float[] target= new float[]{0f, 0f};
        LossFunc MSE= (hs, y) -> {
            float[]res= new float[hs.length];
            for(int i=0; i< res.length; i++) {
                res[i]= hs[i]-y[i];
                res[i]*=2;
            }
            return res;
        };/*
        IMPORTANT: Um den ganzen Rundungsunterschieden evt. vorzubeugen
        cell.hs_t= new float[]{0.0384f, -0.1353f};
        cell.cs_t= new float[]{0.251f, -0.193f};
        cell.backwardPass(MSE, target, new float[]{0.701f, 0.5f}, new float[]{0.218f, 0.606f}, new float[]{0.856f, -0.406f}, new float[]{0.153f, 0.701f});
        */
        cell.backwardPass(TIME_STEP, MSE, target, xt, cell.fActivated[TIME_STEP], cell.iActivated[TIME_STEP], cell.cActivated[TIME_STEP], cell.oActivated[TIME_STEP]);
    }

    @Test
    void testsLstmForwardingC() {
        // TODO: 14.09.2026
        float[][] wf = new float[][]{{0.1f, 0.2f}, {0.3f, 0.1f}};
        float[][] wi = new float[][]{{0.2f, 0.3f}, {0.1f, 0.4f}};
        float[][] wc = new float[][]{{0.4f, 0.2f}, {0.1f, 0.3f}};
        float[][] wo = new float[][]{{0.2f, 0.1f}, {0.3f, 0.2f}};
        float[][] uf = new float[][]{{0.1f, 0.1f}, {0.2f, 0.1f}};
        float[][] ui = new float[][]{{0.1f, 0.2f}, {0.2f, 0.2f}};
        float[][] uc = new float[][]{{0.1f, 0.3f}, {0.3f, 0.1f}};
        float[][] uo = new float[][]{{0.2f, 0.2f}, {0.1f, 0.1f}};
        LstmCell cell= new LstmCell(1, wf, wi, wc, wo, uf, ui, uc, uo);
        float[] xt= new float[]{1f, 0.5f};
        cell.hs_prev= new float[]{0.f, 0.f};
        cell.cs_prev= new float[]{0.f, 0.f};
        int TIME_STEP= 0;
        cell.forwardPass(TIME_STEP, xt);

        Assertions.assertEquals(0.2f, cell.fNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.35f, cell.fNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.35f, cell.iNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.3f, cell.iNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.5f, cell.cNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.25f, cell.cNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.25f, cell.oNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.4f, cell.oNetto[TIME_STEP][1], 1e-02);

        Assertions.assertEquals(-1.387f, cell.fNormalized[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.277f, cell.fNormalized[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.277f, cell.iNormalized[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.277f, cell.iNormalized[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(1.941f, cell.cNormalized[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.832f, cell.cNormalized[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.832f, cell.oNormalized[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.832f, cell.oNormalized[TIME_STEP][1], 1e-02);

        Assertions.assertEquals(-1.387f, cell.fScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.277f, cell.fScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.277f, cell.iScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.277f, cell.iScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(1.941f, cell.cScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.832f, cell.cScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.832f, cell.oScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.832f, cell.oScaledAndMoved[TIME_STEP][1], 1e-02);

        Assertions.assertEquals(0.2f, cell.fActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.569f, cell.fActivated[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.569f, cell.iActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.431f, cell.iActivated[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.96f, cell.cActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.682f, cell.cActivated[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.303f, cell.oActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.697f, cell.oActivated[TIME_STEP][1], 1e-02);

        Assertions.assertEquals(0.546f, cell.cs_t[TIME_STEP][0], 1e-03);
        Assertions.assertEquals(-0.294f, cell.cs_t[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.151f, cell.hs_t[TIME_STEP][0], 1e-03);
        Assertions.assertEquals(-0.199f, cell.hs_t[TIME_STEP][1], 1e-02);

        float[] target= new float[]{0f, 0f};
        LossFunc MSE= (hs, y) -> {
            float[]res= new float[hs.length];
            for(int i=0; i< res.length; i++) {
                res[i]= hs[i]-y[i];
                res[i]*=2;
            }
            return res;
        };/*
        IMPORTANT: Um den ganzen Rundungsunterschieden evt. vorzubeugen
        cell.hs_t= new float[]{0.0384f, -0.1353f};
        cell.cs_t= new float[]{0.251f, -0.193f};
        cell.backwardPass(MSE, target, new float[]{0.701f, 0.5f}, new float[]{0.218f, 0.606f}, new float[]{0.856f, -0.406f}, new float[]{0.153f, 0.701f});
        */
        cell.backwardPass(TIME_STEP, MSE, target, xt, cell.fActivated[TIME_STEP], cell.iActivated[TIME_STEP], cell.cActivated[TIME_STEP], cell.oActivated[TIME_STEP]);
    }
/*
    @Test
    void testsLstmForwardingB() {
        // TODO: 14.09.2026
        float[][] wf = new float[][]{{1f, 1f, 1f}, {0.5f, 0.5f, 0.5f}};
        float[][] wi = new float[][]{{1f, 1f, 1f}, {0.5f, 0.5f, 0.5f}};
        float[][] wc = new float[][]{{1f, 1f, 1f}, {0.5f, 0.5f, 0.5f}};
        float[][] wo = new float[][]{{1f, 1f, 1f}, {0.5f, 0.5f, 0.5f}};
        float[][] uf = new float[][]{{0.0f, 0.0f}, {0.0f, 0.0f}};
        float[][] ui = new float[][]{{-5f, 0.0f}, {0.0f, 1f}};
        float[][] uc = new float[][]{{1f, 0f}, {0f, -1f}};
        float[][] uo = new float[][]{{-4f, -2f}, {1f, 1f}};
        LstmCell cell= new LstmCell(1, wf, wi, wc, wo, uf, ui, uc, uo);
        float[] xt= new float[]{0.5f, -0.2f, 0.1f};
        cell.hs_prev= new float[]{0.1f, 0.1f};
        cell.cs_prev= new float[]{0.1f, 0.1f};
        int TIME_STEP= 0;
        cell.forwardPass(TIME_STEP, xt);

        Assertions.assertEquals(0.4f, cell.fNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.2f, cell.fNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.1f, cell.iNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.3f, cell.iNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.5f, cell.cNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.1f, cell.cNetto[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-0.2f, cell.oNetto[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.4f, cell.oNetto[TIME_STEP][1], 1e-02);

        Assertions.assertEquals(0.85f, cell.fNormalized[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0f, cell.fNormalized[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-1.28f, cell.iNormalized[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.43f, cell.iNormalized[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(1.28f, cell.cNormalized[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.43f, cell.cNormalized[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-1.71f, cell.oNormalized[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.85f, cell.oNormalized[TIME_STEP][1], 1e-02);

        Assertions.assertEquals(0.85f, cell.fScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0f, cell.fScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-1.28f, cell.iScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.43f, cell.iScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(1.28f, cell.cScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.43f, cell.cScaledAndMoved[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(-1.71f, cell.oScaledAndMoved[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.85f, cell.oScaledAndMoved[TIME_STEP][1], 1e-02);

        Assertions.assertEquals(0.701f, cell.fActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.5f, cell.fActivated[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.218f, cell.iActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.606f, cell.iActivated[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.856f, cell.cActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(-0.406f, cell.cActivated[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.153f, cell.oActivated[TIME_STEP][0], 1e-02);
        Assertions.assertEquals(0.701f, cell.oActivated[TIME_STEP][1], 1e-02);

        Assertions.assertEquals(0.2567f, cell.cs_t[TIME_STEP][0], 1e-03);
        Assertions.assertEquals(-0.196f, cell.cs_t[TIME_STEP][1], 1e-02);
        Assertions.assertEquals(0.0384f, cell.hs_t[TIME_STEP][0], 1e-03);
        Assertions.assertEquals(-0.1353f, cell.hs_t[TIME_STEP][1], 1e-02);

        float[] target= new float[]{0f, 0f};
        LossFunc MSE= (hs, y) -> {
            float[]res= new float[hs.length];
            for(int i=0; i< res.length; i++) {
                res[i]= hs[i]-y[i];
                res[i]*=2;
            }
            return res;
        };/*
        IMPORTANT: Um den ganzen Rundungsunterschieden evt. vorzubeugen
        cell.hs_t= new float[]{0.0384f, -0.1353f};
        cell.cs_t= new float[]{0.251f, -0.193f};
        cell.backwardPass(MSE, target, new float[]{0.701f, 0.5f}, new float[]{0.218f, 0.606f}, new float[]{0.856f, -0.406f}, new float[]{0.153f, 0.701f});

        cell.backwardPass(TIME_STEP, MSE, target, xt, cell.fActivated[TIME_STEP], cell.iActivated[TIME_STEP], cell.cActivated[TIME_STEP], cell.oActivated[TIME_STEP]);
    }

    @Test
    void testsLstmBpttLossCalc() {
        float[][] wf = new float[][]{{0.5f, 0.1f}, {-0.2f, 0.4f}};
        float[][] wi = new float[][]{{0.2f, -0.3f}, {0.4f, 0.1f}};
        float[][] wc = new float[][]{{0.3f, 0.2f}, {-0.1f, 0.5f}};
        float[][] wo = new float[][]{{0.1f, 0.4f}, {0.3f, -0.2f}};
        float[][] uf = new float[][]{{0.1f, 0.3f}, {0.2f, -0.1f}};
        float[][] ui = new float[][]{{-0.1f, 0.2f}, {0.3f, 0.4f}};
        float[][] uc = new float[][]{{0.4f, -0.2f}, {0.1f, 0.3f}};
        float[][] uo = new float[][]{{0.2f, 0.1f}, {-0.3f, 0.4f}};
        LstmCell cell= new LstmCell(wf, wi, wc, wo, uf, ui, uc, uo);
        float[] xt= new float[]{1f, -0.5f};
        float[] target= new float[]{0f, 0f};
        cell.hs_prev= new float[]{0.1f, 0.1f};
        cell.cs_prev= new float[]{0.1f, 0.1f};
        cell.forwardPass(xt);
        LossFunc MSE= (hs, y) -> {
            float[]res= new float[hs.length];
            for(int i=0; i< res.length; i++) {
                res[i]= hs[i]-y[i];
                res[i]*=2;
            }
            return res;
        };

        cell.hs_t= new float[]{0.0384f, -0.1353f};
        cell.cs_t= new float[]{0.251f, -0.193f};
        cell.backwardPass(MSE, target, xt, new float[]{0.701f, 0.5f}, new float[]{0.218f, 0.606f}, new float[]{0.856f, -0.406f}, new float[]{0.153f, 0.701f});
// TODO: 15.09.2026
    }

    @Test
    void testsLstmPreActivationNormalization() {
        float[][] wf = new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][] wi = new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][] wc = new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][] wo = new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][] uf = new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[][] ui = new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[][] uc = new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[][] uo = new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        LstmCell cell= new LstmCell(wf, wi, wc, wo, uf, ui, uc, uo);
        float[]combinedLinearProjection= new float[]{0.4f, 0.2f, -0.1f, 0.3f, 0.5f, 0.1f, -0.2f, 0.4f};
        float[]aHat= cell.preActivationNormalization(combinedLinearProjection);
        Assertions.assertEquals(0.85f, aHat[0], 1e-02);
        Assertions.assertEquals(0.0f, aHat[1], 1e-02);
        Assertions.assertEquals(-1.28f, aHat[2], 1e-02);
        Assertions.assertEquals(0.43f, aHat[3], 1e-02);
        Assertions.assertEquals(1.28f, aHat[4], 1e-02);
        Assertions.assertEquals(-0.43f, aHat[5], 1e-02);
        Assertions.assertEquals(-1.71f, aHat[6], 1e-02);
        Assertions.assertEquals(0.85f, aHat[7], 1e-02);
    }

    @Test
    void testsLstmForwarding() {
        // TODO: 14.09.2026
        float[][] wf = new float[][]{{0.5f, 0.1f}, {-0.2f, 0.4f}};
        float[][] wi = new float[][]{{0.2f, -0.3f}, {0.4f, 0.1f}};
        float[][] wc = new float[][]{{0.3f, 0.2f}, {-0.1f, 0.5f}};
        float[][] wo = new float[][]{{0.1f, 0.4f}, {0.3f, -0.2f}};
        float[][] uf = new float[][]{{0.1f, 0.3f}, {0.2f, -0.1f}};
        float[][] ui = new float[][]{{-0.1f, 0.2f}, {0.3f, 0.4f}};
        float[][] uc = new float[][]{{0.4f, -0.2f}, {0.1f, 0.3f}};
        float[][] uo = new float[][]{{0.2f, 0.1f}, {-0.3f, 0.4f}};
        LstmCell cell= new LstmCell(wf, wi, wc, wo, uf, ui, uc, uo);
        float[] xt= new float[]{1f, -0.5f};
        cell.hs_prev= new float[]{0.2f, 0.1f};
        cell.cs_prev= new float[]{0.5f, 0.0f};
        cell.forwardPass(xt);
        Assertions.assertEquals(0.5f, cell.fNetto[0], 1e-02);
        Assertions.assertEquals(-0.37f, cell.fNetto[1], 1e-02);
        Assertions.assertEquals(0.35f, cell.iNetto[0], 1e-02);
        Assertions.assertEquals(0.45f, cell.iNetto[1], 1e-02);
        Assertions.assertEquals(0.26f, cell.cNetto[0], 1e-02);
        Assertions.assertEquals(-0.3f, cell.cNetto[1], 1e-02);
        Assertions.assertEquals(-0.05f, cell.oNetto[0], 1e-02);
        Assertions.assertEquals(0.38f, cell.oNetto[1], 1e-02);

        Assertions.assertEquals(1f, cell.fNormalized[0], 1e-02);
        Assertions.assertEquals(-1f, cell.fNormalized[1], 1e-02);
        Assertions.assertEquals(-1f, cell.iNormalized[0], 1e-02);
        Assertions.assertEquals(1f, cell.iNormalized[1], 1e-02);
        Assertions.assertEquals(1f, cell.cNormalized[0], 1e-02);
        Assertions.assertEquals(-1f, cell.cNormalized[1], 1e-02);
        Assertions.assertEquals(-1f, cell.oNormalized[0], 1e-02);
        Assertions.assertEquals(1f, cell.oNormalized[1], 1e-02);

        Assertions.assertEquals(1f, cell.fScaledAndMoved[0], 1e-02);
        Assertions.assertEquals(-1f, cell.fScaledAndMoved[1], 1e-02);
        Assertions.assertEquals(-1f, cell.iScaledAndMoved[0], 1e-02);
        Assertions.assertEquals(1f, cell.iScaledAndMoved[1], 1e-02);
        Assertions.assertEquals(1f, cell.cScaledAndMoved[0], 1e-02);
        Assertions.assertEquals(-1f, cell.cScaledAndMoved[1], 1e-02);
        Assertions.assertEquals(-1f, cell.oScaledAndMoved[0], 1e-02);
        Assertions.assertEquals(1f, cell.oScaledAndMoved[1], 1e-02);

        Assertions.assertEquals(0.731f, cell.fActivated[0], 1e-02);
        Assertions.assertEquals(0.269f, cell.fActivated[1], 1e-02);
        Assertions.assertEquals(0.269f, cell.iActivated[0], 1e-02);
        Assertions.assertEquals(0.731f, cell.iActivated[1], 1e-02);
        Assertions.assertEquals(0.762f, cell.cActivated[0], 1e-02);
        Assertions.assertEquals(-0.762f, cell.cActivated[1], 1e-02);
        Assertions.assertEquals(0.269f, cell.oActivated[0], 1e-02);
        Assertions.assertEquals(0.731f, cell.oActivated[1], 1e-02);

        Assertions.assertEquals(0.5705f, cell.cs_t[0], 1e-03);
        Assertions.assertEquals(-0.557f, cell.cs_t[1], 1e-03);
        Assertions.assertEquals(0.138f, cell.hs_t[0], 1e-03);
        Assertions.assertEquals(-0.37f, cell.hs_t[1], 1e-02);
    }

 */
}
