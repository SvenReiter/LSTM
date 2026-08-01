package lstm2608010711_BerechneteResultateVonJedemTimestepSindWeggespeichert;

public class Lstm {
    /*TODO
        1. Die Eingangsschicht (Input Layer / Embedding Layer)
            Bestehend aus: Samples, Time Steps, Features
                Ein Sample ist eine einzelne, in sich geschlossene Beobachtung oder ein einzelnes Datenobjekt in Ihrem Datensatz.
                    Kurz: Die Anzahl der Zeilen bzw. Datensätze in einem/r klassischen Machine-Learning-Split/-Datenaufteilung.
                Time Steps: Um welchen TimeStep es sich handelt
                    Kurz: Die "Zeitachse" oder Länge der Kette, die das LSTM Schritt für Schritt durchläuft.
                Features: Features sind die einzelnen Messwerte oder Eigenschaften, die Sie an jedem einzelnen Zeitschritt (Time Step) aufzeichnen.
                    Kurz: Die Anzahl der Eingangsvariablen pro Zeitschritt.
        2. LSTM Layer {LstmCell die iterativ oder rekursiv die Werte/Gewichte anpasst/berechnet/verrechnet}
        3. Batch-Normalization-Layer (es gibt 2 Alternativen Sequence-to-Sequence-Szenario und Many-to-One-Szenario)
           Das Sequence-to-Sequence-Szenario implementiert meine bisherige BatchNormalizationLayer-Implementierung.
        4. Die Ausgangsschicht (Output Layer / Dense + Softmax)
            Der Dense-Layer ändert die Anzahl (Dimension): Er schrumpft (oder erweitert) die Datenmenge auf die exakte Anzahl Ihrer Zielklassen.
            Das Ergebnis sind rohe Punkte für jede Klasse (z. B. Klasse A: 4.5, Klasse B: 1.2, Klasse C: -0.8).
            Das Softmax ändert das Format: Es ändert nichts mehr an der Anzahl der Klassen,
            sondern wandelt diese rohen Punkte nur noch in lesbare Prozentwerte um (z. B. 92%, 6%, 2%).
     */
    LstmCell cell;
    public Lstm(float[] bf, float[] bi, float[] bc, float[] bo,
                float[][] wf, float[][] wi, float[][] wc, float[][] wo,
                float[][] uf, float[][] ui, float[][] uc, float[][] uo,
                float[] target) {
        cell= new LstmCell(bf, bi, bc, bo, wf, wi, wc, wo, uf, ui, uc, uo, target); //Aus technischer Sicht existiert in einem neuronalen Netzwerk meistens nur ein einziges physisches LSTM-Zell-Objekt,
    }

    public void forwarding(int TIMESTEP, float[]...input) {
        for(int i=0; i< input.length; i++) {
            cell.forwarding(TIMESTEP, input[i]);     //das nacheinander für alle Zeitschritte wiederverwendet wird.
        }
    }

    public void bptt(float[]y, float[]...x) {
        cell.bptt(y, x);
    }
}
