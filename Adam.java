package lstm2609251410;

public class Adam {
    float ALPHA= 0.001f;
    float BETA1= 0.9f;
    float BETA2= 0.999f;
    float ETA= 0.001f;
    float EPSILON= (float)1e-8;
    float[][][] m;
    float[][][] v;

    public Adam(int timesteps, float[][]delL_delW, float[][]oldWeight) {
        int PRE_ELM= 1;
        m= new float[PRE_ELM+timesteps][delL_delW.length][delL_delW[0].length];
        v= new float[PRE_ELM+timesteps][delL_delW.length][delL_delW[0].length];
        for(int l=0; l<delL_delW.length; l++) {
            for (int c = 0; c < delL_delW[l].length; c++) {
                m[0][l][c]= 0f; //Da 1. Optimierungsschritt(t=1), gehen wir davon aus, vorherige Momente starten bei 0.0
                v[0][l][c]= 0f; //Da 1. Optimierungsschritt(t=1), gehen wir davon aus, vorherige Momente starten bei 0.0
            }
        }
    }

    /**
     *
     * @param timestep
     * @param delL_delW calculated gradients
     * @param oldWeight original weight matrix
     * @return optimized weight matrix */
    public float[][] calc(int timestep, float[][]delL_delW, float[][]oldWeight) {
        // done: 25.09.2026 Schritt 2: Berechnung des 1. Moments (m1)
        for(int l=0; l<delL_delW.length; l++) {
            for (int c = 0; c < delL_delW[l].length; c++) {
                m[timestep][l][c]= BETA1* m[timestep-1][l][c]+ (1-BETA1)* delL_delW[l][c];
            }
        }

        // done: 25.09.2026 Schritt 3: Berechnung des 2. Moments (v1)
        for(int l=0; l<delL_delW.length; l++) {
            for (int c = 0; c < delL_delW[l].length; c++) {
                v[timestep][l][c]= BETA2* v[timestep-1][l][c]+ (1-BETA2)* (float)Math.pow(delL_delW[l][c], 2);
            }
        }

        // done: 25.09.2026 Schritt 4: Bias-Korrektur (Bias Correction)
        float[][][] mHat= new float[m.length][m[0].length][m[0][0].length];
        float[][][] vHat= new float[v.length][v[0].length][v[0][0].length];
        for(int l=0; l<delL_delW.length; l++) {
            for (int c = 0; c < delL_delW[l].length; c++) {
                mHat[timestep][l][c]= m[timestep][l][c]/(1-BETA1);
                vHat[timestep][l][c]= v[timestep][l][c]/(1-BETA2);
            }
        }

        // done: 25.09.2026 Schritt 5: Parameter-Update
        float[][]newWeight= new float[delL_delW.length][delL_delW[0].length];
        for(int l=0; l<delL_delW.length; l++) {
            for(int c=0; c<delL_delW[l].length; c++) {
                newWeight[l][c]= oldWeight[l][c]- ETA* (mHat[timestep][l][c]/ ((float)Math.sqrt(vHat[timestep][l][c])+ EPSILON));
            }
        }
        return newWeight;
    }
}
