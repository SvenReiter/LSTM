package lstm2604131306;

/**
 * Eine LSTM Batch Normalization Layer sorgt dafür,
 * dass die Daten innerhalb eines neuronalen Netzwerks (LSTM) während des Trainings stabil bleiben.
 *     Sie "normalisiert" die Werte: Sie berechnet Mittelwert und Varianz für einen kleinen Teil der Daten (Batch) und
 *                                   bringt die Werte in einen ähnlichen Bereich(Mittelwert 0,Standardabweichung 1).
 *     Sie stabilisiert das Lernen: Dadurch wird verhindert, dass die Werte im Netzwerk zu groß oder zu klein werden,
 *                                  was das Training oft instabil macht.
 *     Sie beschleunigt das Training: Das Netzwerk konvergiert schneller, das heißt,
 *                                    es lernt in kürzerer Zeit bessere Ergebnisse.
 *     Sie reduziert den "Internal Covariate Shift": Sie verhindert, dass sich die Verteilung der Eingaben für eine
 *                                                   Schicht ständig ändert, weil vorherige Schicht noch am Lernen ist.*/
public class BatchNormalizationLayer {
    private float[][] normalization;

    /**
     *Backpropagation pro Feature-Vektor
     * @param m         Anzahl Werte pro Feature bzw der "Mini-Batch"
     * @param x         Feature-Vektor
     * @param beta      (Verschiebung/Bias): Verschiebt den Mittelwert. Übernimmt die Rolle des klassischen "Bias"-Terms,
     *                                      der durch die Subtraktion von bei der Normalisierung verloren gegangen ist.
     * @param gamma     (Skalierung): Steuert wie "breit" die Verteilung ist. Es bestimmt, wie stark die Nichtlinearität
     *                               der folgenden Aktivierungsfunktion (z. B. ReLU oder Tanh) ausgenutzt wird.
     * @param epsilon   eine winzige positive Konstante (ein sogenannter Smoothing Term)
     * @param dY        eingehender Gradient
     * @return          dZ[][] Angepasste Gradient.
     *                  Diese Gradienten zeigen, wie die Gate-Aktivierungen der vorherigen Mini-Batch angepasst werden
     *                  müssen, um den Fehler zu reduzieren. */
    public float[][] backpropagate(int m, float[]x, float beta, float gamma, float epsilon, float[]dY) {
        float[]averages= getAverage(x);
        float[]variances= getVariance(new float[][]{x}, averages);
        float[][]normalizations= getNormalization(new float[][]{x}, averages, variances, epsilon);
        float[][]transformations= getTransformations(normalizations, beta, gamma);
        float[][]dZ= new float[dY.length][m];                                                                           // Ziel: die Gradienten bezüglich des Inputs (dZ) berechnen
        float standardabweichung= (float)Math.sqrt(variances[0]+epsilon);
        float dZscaling= gamma/(m*standardabweichung);  //Vorfaktor
        float dBeta= 0, dGamma=0;
        for(int i=0; i<dY.length; i++) {
            dBeta+=dY[i];
            dGamma+=dY[i]*normalizations[0][i];
        }
        for (float[] miniBatch : dZ) {
            for (int i = 0; i < m; i++) {
                miniBatch[i] = m * dY[i] - dBeta - transformations[0][i] * dGamma;
                miniBatch[i] *= dZscaling;
            }
        }
        return dZ;
    }

    public float[][] getTransformations(float[][]normalizations, float beta, float gamma) {
        float[][]transformations= new float[normalizations.length][normalizations[0].length];
        for(int line=0; line<transformations.length; line++) {
            for(int col=0; col<transformations[line].length; col++) {
                transformations[line][col]= normalizations[line][col]*gamma;                                            //Skalierung und...
                transformations[line][col]+= beta;                                                                      //...Verschiebung
            }
        }
        return transformations;
    }

    private float[] calcAverage(float[]... activations) {
        float[] averages= new float[activations.length];
        for(int i=0; i<activations.length; i++) {
            for(float f: activations[i]) {
                averages[i]+=f;
            }
            averages[i]/=activations[i].length;
        }
        return averages;
    }

    private float[] calcVariance(float[][]activations, float[]averages) {
        float[] variances= new float[activations.length];
        for(int l=0; l<activations.length; l++) {
            for(int c=0; c<activations[l].length; c++) {
                variances[l]+= Math.pow((activations[l][c]-averages[l]), 2);
            }
            variances[l]/=activations[l].length;
        }
        return variances;
    }

    private float[][] calcNormalization(float[][]activations, float[]averages, float[]variances, float epsilon) {       //ist Normalisierung
        float[][] normalized= new float[activations.length][activations[0].length];
        for(int l=0; l<activations.length; l++) {
            for(int c=0; c<activations[l].length; c++) {
                normalized[l][c]+= (activations[l][c]-averages[l]);
                normalized[l][c]/= Math.sqrt((variances[l]+epsilon));
            }
        }
        return normalized;
    }

    public void forward(float epsilon, float[]... activations) {
        float[] averages= calcAverage(activations);
        float[] variances= calcVariance(activations, averages);
        normalization= calcNormalization(activations, averages, variances, epsilon);
    }

    public BatchNormalizationLayer(float epsilon, float[]... activations) {
        forward(epsilon, activations);
    }
    public BatchNormalizationLayer(int NUMBER_CELLS_PER_CHAIN,int NUMBER_FEATURES) {
        normalization= new float[NUMBER_CELLS_PER_CHAIN][NUMBER_FEATURES];
    }
    public BatchNormalizationLayer(){}

    public float[] getAverage(float[]... activations) {
        return calcAverage(activations);
    }
    public float[] getVariance(float[][]activations, float[]averages) {
        return calcVariance(activations, averages);
    }
    public float[][] getNormalization(float[][]activations, float[]averages, float[]variances, float epsilon) {
        return calcNormalization(activations, averages, variances, epsilon);
    }
    public float[][] getNormalization() {
        return normalization;
    }
    public float[][] scaleAndShift(float[][]normalization, float gamma, float beta) {
        for(int z = 0; z< normalization.length; z++) {
            for(int i = 0; i< normalization[z].length; i++) {
                normalization[z][i]*=gamma;
                normalization[z][i]+=beta;
            }
        }
        return normalization;
    }
}
