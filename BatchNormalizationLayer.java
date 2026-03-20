package lstm2603200945;

/**
 * To standardizes the inputs to a layer for each mini-batch, which helps to:
 *     Accelerate the training process.
 *     Improve model stability.
 *     Reduce sensitivity to initialization.
 *     Act as a regularizer, often reducing the need for dropout.*/
public class BatchNormalizationLayer {

    /*todo
        In erster LstmCellRow wird ein Input(X[t]) pro LstmCell/Zeitschritt empfangen.
        Jede LstmCell dieser LstmCellRow kommuniziert ihren resultierenden cellState[t] und hiddenState[t] an die
            folgende LstmCell.
        Des Weiteren kommuniziert jede LstmCell dieser LstmCellRow ihren resultierenden hiddenState[t] an das
            verantwortliche BatchNormalizationLayer.
        Das BatchNormalizationLayer kommuniziert seine standard-normalverteilten hiddenState-Werte der LstmCells von
            der vorherigen Row als Input an die LstmCells der nächste Row.
        *
        Im Sinne der Backpropagation
            Der BatchNormalizationLayer verändert die Werte ja nicht, der Layer bestimmt die Standardnormalverteilung.
            Weil diese standardnormalverteilten Werte bloß ungewichtet an nächste Schicht weitergegeben werden,
            bedarf es hierbei auch keiner weiteren Berechnung für das Gradientenverfahren.
     */
    private float[][] zValues;

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
                variances[l]+= Math.pow((activations[l][c]-averages[l]), activations[l].length);
            }
            variances[l]/=activations[l].length;
        }
        return variances;
    }

    private float[][] calcZValue(float[][]activations, float[]averages, float[]variances) {
        float[][] zValues= new float[activations.length][activations[0].length];
        for(int l=0; l<activations.length; l++) {
            for(int c=0; c<activations[l].length; c++) {
                zValues[l][c]+= (activations[l][c]-averages[l]);
                zValues[l][c]/= Math.sqrt(variances[l]);
            }
        }
        return zValues;
    }

    public BatchNormalizationLayer(float[]... activations) {
        float[] averages= calcAverage(activations);
        float[] variances= calcVariance(activations, averages);
        zValues= calcZValue(activations, averages, variances);
    }

    public float[] getAverage(float[]... activations) {
        return calcAverage(activations);
    }
    public float[] getVariance(float[][]activations, float[]averages) {
        return calcVariance(activations, averages);
    }
    public float[][] getZValue(float[][]activations, float[]averages, float[]variances) {
        return calcZValue(activations, averages, variances);
    }
    public float[][] getZValue() {
        return zValues;
    }
}
