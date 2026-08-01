package lstm2608010711_BerechneteResultateVonJedemTimestepSindWeggespeichert;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class tests_LSTM {

    @Test
    void testsLstmHoldingMultipleHsCsGateActivationResultsAcrossEveryTimestep() {
        float[] bf = new float[]{0.1f, 0.1f}, bi = new float[]{0.1f, 0.1f}, bc = new float[]{0.1f, 0.1f}, bo = new float[]{0.1f, 0.1f};
        float[][] wf = new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][] wi = new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][] wc = new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][] wo = new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][] uf = new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[][] ui = new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[][] uc = new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[][] uo = new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[] target = new float[]{0.5f, 0.5f};
        Lstm lstm = new Lstm(bf, bi, bc, bo, wf, wi, wc, wo, uf, ui, uc, uo, target);
        lstm.cell.cs.add(new float[]{0f, 0f});
        lstm.cell.hs.add(new float[]{0f, 0f});
        assertEquals(lstm.cell.matMul(uf, new float[]{0.2153f, 0.0275f})[0], 0.04581f);
        assertEquals(lstm.cell.matMul(uf, new float[]{0.2153f, 0.0275f})[1], 0.02703f);
        float[]x1= new float[]{1.0f, 0f};
        float[]x2= new float[]{0f, 1.0f};
        float[]x3= new float[]{1.0f, 1.0f};

        int TIME_STEP= 0;
        lstm.forwarding(TIME_STEP, x1);
        assertEquals(lstm.cell.cs.get(TIME_STEP)[0], 0.3467f, 0.001);
        assertEquals(lstm.cell.cs.get(TIME_STEP)[1], 0.0523f, 0.001);
        assertEquals(lstm.cell.hs.get(TIME_STEP)[0], 0.2153f, 0.001);
        assertEquals(lstm.cell.hs.get(TIME_STEP)[1], 0.0275f, 0.001);
        assertEquals(lstm.cell.forgetGateNetto.get(TIME_STEP)[0], 0.6f, 0.001);
        assertEquals(lstm.cell.forgetGateNetto.get(TIME_STEP)[1], 0.1f, 0.001);
        assertEquals(lstm.cell.inputGateNetto.get(TIME_STEP)[0], 0.6f, 0.001);
        assertEquals(lstm.cell.inputGateNetto.get(TIME_STEP)[1], 0.1f, 0.001);
        assertEquals(lstm.cell.candidateGateNetto.get(TIME_STEP)[0], 0.6f, 0.001);
        assertEquals(lstm.cell.candidateGateNetto.get(TIME_STEP)[1], 0.1f, 0.001);
        assertEquals(lstm.cell.outputGateNetto.get(TIME_STEP)[0], 0.6f, 0.001);
        assertEquals(lstm.cell.outputGateNetto.get(TIME_STEP)[1], 0.1f, 0.001);
        assertEquals(lstm.cell.forgetGateActivation.get(TIME_STEP)[0], 0.6457f, 0.001);
        assertEquals(lstm.cell.forgetGateActivation.get(TIME_STEP)[1], 0.525f, 0.001);
        assertEquals(lstm.cell.inputGateActivation.get(TIME_STEP)[0], 0.6457f, 0.001);
        assertEquals(lstm.cell.inputGateActivation.get(TIME_STEP)[1], 0.525f, 0.001);
        assertEquals(lstm.cell.candidateGateActivation.get(TIME_STEP)[0], 0.537f, 0.001);
        assertEquals(lstm.cell.candidateGateActivation.get(TIME_STEP)[1], 0.0997f, 0.001);
        assertEquals(lstm.cell.outputGateActivation.get(TIME_STEP)[0], 0.6457f, 0.001);
        assertEquals(lstm.cell.outputGateActivation.get(TIME_STEP)[1], 0.525f, 0.001);

        TIME_STEP++;
        lstm.forwarding(TIME_STEP, x2);
        assertEquals(lstm.cell.cs.get(TIME_STEP)[0], 0.2637f, 0.0001);
        assertEquals(lstm.cell.cs.get(TIME_STEP)[1], 0.3964f, 0.001);
        assertEquals(lstm.cell.hs.get(TIME_STEP)[0], 0.1383f, 0.0001);      // DONE: 31.07.2026 Fehler untersuchen
        assertEquals(lstm.cell.hs.get(TIME_STEP)[1], 0.246f, 0.001);      // DONE: 31.07.2026 Fehler untersuchen
        assertEquals(lstm.cell.forgetGateNetto.get(TIME_STEP)[0], 0.1458f, 0.0001);
        assertEquals(lstm.cell.forgetGateNetto.get(TIME_STEP)[1], 0.627f, 0.001);
        assertEquals(lstm.cell.inputGateNetto.get(TIME_STEP)[0], 0.1458f, 0.0001);
        assertEquals(lstm.cell.inputGateNetto.get(TIME_STEP)[1], 0.627f, 0.001);
        assertEquals(lstm.cell.candidateGateNetto.get(TIME_STEP)[0], 0.1458f, 0.0001);
        assertEquals(lstm.cell.candidateGateNetto.get(TIME_STEP)[1], 0.627f, 0.001);
        assertEquals(lstm.cell.outputGateNetto.get(TIME_STEP)[0], 0.1458f, 0.0001);
        assertEquals(lstm.cell.outputGateNetto.get(TIME_STEP)[1], 0.627f, 0.001);
        assertEquals(lstm.cell.forgetGateActivation.get(TIME_STEP)[0], 0.5364f, 0.0001);
        assertEquals(lstm.cell.forgetGateActivation.get(TIME_STEP)[1], 0.6518f, 0.0001);
        assertEquals(lstm.cell.inputGateActivation.get(TIME_STEP)[0], 0.5364f, 0.0001);
        assertEquals(lstm.cell.inputGateActivation.get(TIME_STEP)[1], 0.6518f, 0.0001);
        assertEquals(lstm.cell.candidateGateActivation.get(TIME_STEP)[0], 0.1448f, 0.0001);
        assertEquals(lstm.cell.candidateGateActivation.get(TIME_STEP)[1], 0.556f, 0.001);
        assertEquals(lstm.cell.outputGateActivation.get(TIME_STEP)[0], 0.5364f, 0.0001);
        assertEquals(lstm.cell.outputGateActivation.get(TIME_STEP)[1], 0.6518f, 0.0001);

        TIME_STEP++;
        lstm.forwarding(TIME_STEP, x3);
        assertEquals(lstm.cell.cs.get(TIME_STEP)[0], 0.5506f, 0.001);
        assertEquals(lstm.cell.cs.get(TIME_STEP)[1], 0.6444f, 0.001);
        assertEquals(lstm.cell.hs.get(TIME_STEP)[0], 0.33f, 0.01);
        assertEquals(lstm.cell.hs.get(TIME_STEP)[1], 0.375f, 0.001);      // DONE: 31.07.2026 Fehler untersuchen
        assertEquals(lstm.cell.forgetGateNetto.get(TIME_STEP)[0], 0.6522f, 0.0001);
        assertEquals(lstm.cell.forgetGateNetto.get(TIME_STEP)[1], 0.66297f, 0.00001);
        assertEquals(lstm.cell.inputGateNetto.get(TIME_STEP)[0], 0.6522f, 0.0001);
        assertEquals(lstm.cell.inputGateNetto.get(TIME_STEP)[1], 0.66297f, 0.00001);
        assertEquals(lstm.cell.candidateGateNetto.get(TIME_STEP)[0], 0.6522f, 0.0001);
        assertEquals(lstm.cell.candidateGateNetto.get(TIME_STEP)[1], 0.66297f, 0.00001);
        assertEquals(lstm.cell.outputGateNetto.get(TIME_STEP)[0], 0.6522, 0.0001);
        assertEquals(lstm.cell.outputGateNetto.get(TIME_STEP)[1], 0.66297f, 0.00001);
        assertEquals(lstm.cell.forgetGateActivation.get(TIME_STEP)[0], 0.658f, 0.001);
        assertEquals(lstm.cell.forgetGateActivation.get(TIME_STEP)[1], 0.66f, 0.01);
        assertEquals(lstm.cell.inputGateActivation.get(TIME_STEP)[0], 0.658f, 0.001);
        assertEquals(lstm.cell.inputGateActivation.get(TIME_STEP)[1], 0.66f, 0.01);
        assertEquals(lstm.cell.candidateGateActivation.get(TIME_STEP)[0], 0.5731f, 0.0001);
        assertEquals(lstm.cell.candidateGateActivation.get(TIME_STEP)[1], 0.58f, 0.01);
        assertEquals(lstm.cell.outputGateActivation.get(TIME_STEP)[0], 0.658f, 0.001);
        assertEquals(lstm.cell.outputGateActivation.get(TIME_STEP)[1], 0.66f, 0.01);
        /* TODO: 30.07.2026
                nach dem Prinzip der Umgestaltung/Modifizierung von forgetGateNetto müssen auch die Sequenzen
                inputGateNetto, candidateGateNetto, outputGateNetto, forgetGateActivation, inputGateActivation,
                  candidateGateActivation, outputGateActivation, hs, cs
                angepasst werden und die Notwendigkeit von hsPre und csPre entfällt,
                weil auf diese Werte anderweitig zugegriffen werden kann. Was im Code noch angepasst werden muss!!!
         */
        System.out.println();
    }

    @Test
    void testsLstmMatMul() {
        float[] bf = new float[]{0.1f, 0.1f}, bi = new float[]{0.1f, 0.1f}, bc = new float[]{0.1f, 0.1f}, bo = new float[]{0.1f, 0.1f};
        float[][] wf = new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][] wi = new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][] wc = new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][] wo = new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][] uf = new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[][] ui = new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[][] uc = new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[][] uo = new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[] target = new float[]{0.5f, 0.5f};
        Lstm lstm = new Lstm(bf, bi, bc, bo, wf, wi, wc, wo, uf, ui, uc, uo, target);
        lstm.cell.cs.add(new float[]{0f, 0f});
        lstm.cell.hs.add(new float[]{0f, 0f});
        assertEquals(lstm.cell.matMul(uf, new float[]{0.2153f, 0.0275f})[0], 0.04581f);
        assertEquals(lstm.cell.matMul(uf, new float[]{0.2153f, 0.0275f})[1], 0.02703f);
        float[]x1= new float[]{1.0f, 0f};
        int TIME_STEP= 0;
        lstm.forwarding(TIME_STEP, x1);
        assertEquals(lstm.cell.cs.get(TIME_STEP)[0], 0.3467f, 0.001);
        assertEquals(lstm.cell.cs.get(TIME_STEP)[1], 0.0523f, 0.001);
        assertEquals(lstm.cell.hs.get(TIME_STEP)[0], 0.2153f, 0.001);
        assertEquals(lstm.cell.hs.get(TIME_STEP)[1], 0.0275f, 0.001);
    }

    @Test
    void testsLstmA() {
        float[]bf=new float[]{0.1f,0.1f},bi=new float[]{0.1f,0.1f},bc=new float[]{0.1f,0.1f},bo=new float[]{0.1f,0.1f};
        float[][]wf= new float[][]{{0.5f, 0f}, {0f, 0.5f}}; float[][]wi= new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][]wc= new float[][]{{0.5f, 0f}, {0f, 0.5f}}; float[][]wo= new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][]uf= new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}}; float[][]ui= new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[][]uc= new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}}; float[][]uo= new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[]target= new float[]{0.5f, 0.5f};
        Lstm lstm= new Lstm(bf,bi,bc,bo, wf,wi,wc,wo, uf,ui,uc,uo, target);
        int TIME_STEP= 0;
        lstm.cell.cs.set(TIME_STEP, new float[]{0f, 0f});
        lstm.cell.hs.set(TIME_STEP, new float[]{0f, 0f});
        float[]x1= new float[]{1.0f, 0f};
        float[]x2= new float[]{0f, 1.0f};
        float[]x3= new float[]{1.0f, 1.0f};
        lstm.forwarding(TIME_STEP, x1);
        assertEquals(lstm.cell.csPre[0], 0f, 0.001);
        assertEquals(lstm.cell.csPre[1], 0f, 0.001);
        assertEquals(lstm.cell.hsPre[0], 0f, 0.001);
        assertEquals(lstm.cell.hsPre[1], 0f, 0.001);
        assertEquals(lstm.cell.cs.get(TIME_STEP)[0], 0.3467f, 0.001);
        assertEquals(lstm.cell.cs.get(TIME_STEP)[1], 0.0523f, 0.001);
        assertEquals(lstm.cell.hs.get(TIME_STEP)[0], 0.2153f, 0.001);
        assertEquals(lstm.cell.hs.get(TIME_STEP)[1], 0.0275f, 0.001);

        TIME_STEP= 1;
        lstm= new Lstm(bf,bi,bc,bo, wf,wi,wc,wo, uf,ui,uc,uo, target);
        lstm.cell.cs.set(TIME_STEP, new float[]{0f, 0f});
        lstm.cell.hs.set(TIME_STEP, new float[]{0f, 0f});
        lstm.forwarding(TIME_STEP, x1, x2);
        assertEquals(lstm.cell.csPre[0], 0.3467f, 0.001);
        assertEquals(lstm.cell.csPre[1], 0.0523f, 0.001);
        assertEquals(lstm.cell.hsPre[0], 0.2153f, 0.001);
        assertEquals(lstm.cell.hsPre[1], 0.0275f, 0.001);
        assertEquals(lstm.cell.cs.get(TIME_STEP)[0], 0.2637f, 0.001);
        assertEquals(lstm.cell.cs.get(TIME_STEP)[1], 0.3964f, 0.001);
        assertEquals(lstm.cell.hs.get(TIME_STEP)[0], 0.1383f, 0.001);
        assertEquals(lstm.cell.hs.get(TIME_STEP)[1], 0.2457f, 0.001);

        TIME_STEP= 2;
        lstm= new Lstm(bf,bi,bc,bo, wf,wi,wc,wo, uf,ui,uc,uo, target);
        lstm.cell.cs.set(TIME_STEP, new float[]{0f, 0f});
        lstm.cell.hs.set(TIME_STEP, new float[]{0f, 0f});
        lstm.forwarding(TIME_STEP, x1, x2, x3);
        assertEquals(lstm.cell.csPre[0], 0.2637f, 0.001);
        assertEquals(lstm.cell.csPre[1], 0.3964f, 0.001);
        assertEquals(lstm.cell.hsPre[0], 0.1383f, 0.001);
        assertEquals(lstm.cell.hsPre[1], 0.2457f, 0.001);
        assertEquals(lstm.cell.cs.get(TIME_STEP)[0], 0.5506f, 0.001);
        assertEquals(lstm.cell.cs.get(TIME_STEP)[1], 0.6444f, 0.001);
        assertEquals(lstm.cell.hs.get(TIME_STEP)[0], 0.33f, 0.001);
        assertEquals(lstm.cell.hs.get(TIME_STEP)[1], 0.375f, 0.001);

        lstm.bptt(new float[]{0.5f, 0.5f}, x1, x2, x3);
    }
}
