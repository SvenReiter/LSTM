package lstm2609251410;

import java.util.Arrays;

public class LstmCell {
    final byte ANZAHL_GATES;
    final byte ANZAHL_FEATURES;
    final double EPSILON= 1e-05;
    final Func SIGMOID= x-> (float)(1 / (1 + Math.exp(-x[0])));
    final Func TANH= x-> (float)((Math.exp(x[0])-Math.exp(-x[0])) / (Math.exp(x[0])+Math.exp(-x[0])));
    private float[][] wf, wi, wc, wo;
    private float[][] uf, ui, uc, uo;
    float[][]fNetto;
    float[][]iNetto;
    float[][]cNetto;
    float[][]oNetto;
    float[][]fNormalized;
    float[][]iNormalized;
    float[][]cNormalized;
    float[][]oNormalized;
    float[][]fScaledAndMoved;
    float[][]iScaledAndMoved;
    float[][]cScaledAndMoved;
    float[][]oScaledAndMoved;
    float[][]fActivated;
    float[][]iActivated;
    float[][]cActivated;
    float[][]oActivated;
    float[][][]delL_delWf;
    float[][][]delL_delWi;
    float[][][]delL_delWc;
    float[][][]delL_delWo;
    float[][][]delL_delUf;
    float[][][]delL_delUi;
    float[][][]delL_delUc;
    float[][][]delL_delUo;
    float[][]hs_t,cs_t;
    float[]hs_prev,cs_prev;
    float fBeta, fGamma;
    float iBeta, iGamma;
    float cBeta, cGamma;
    float oBeta, oGamma;

    double standardabweichung;
    LstmCell(int timesteps,
             float[][] wf, float[][] wi, float[][] wc, float[][] wo,
             float[][] uf, float[][] ui, float[][] uc, float[][] uo) {
        this.wf= wf; this.wi= wi; this.wc= wc; this.wo= wo;
        this.uf= uf; this.ui= ui; this.uc= uc; this.uo= uo;
        ANZAHL_GATES= 4;
        ANZAHL_FEATURES= 2;
        fGamma=1; fBeta=0;
        iGamma=1; iBeta=0;
        cGamma=1; cBeta=0;
        oGamma=1; oBeta=0;
        fNetto= new float[timesteps][];
        iNetto= new float[timesteps][];
        cNetto= new float[timesteps][];
        oNetto= new float[timesteps][];
        fNormalized= new float[timesteps][];
        iNormalized= new float[timesteps][];
        cNormalized= new float[timesteps][];
        oNormalized= new float[timesteps][];
        fScaledAndMoved= new float[timesteps][];
        iScaledAndMoved= new float[timesteps][];
        cScaledAndMoved= new float[timesteps][];
        oScaledAndMoved= new float[timesteps][];
        fActivated= new float[timesteps][];
        iActivated= new float[timesteps][];
        cActivated= new float[timesteps][];
        oActivated= new float[timesteps][];
        delL_delWf= new float[timesteps][][];
        delL_delWi= new float[timesteps][][];
        delL_delWc= new float[timesteps][][];
        delL_delWo= new float[timesteps][][];
        delL_delUf= new float[timesteps][][];
        delL_delUi= new float[timesteps][][];
        delL_delUc= new float[timesteps][][];
        delL_delUo= new float[timesteps][][];
        cs_t= new float[timesteps][];
        hs_t= new float[timesteps][];
    }

    /**
     * @param timestep
     * @param x_t the current timesteps inputFeature
     * @post Resulting values of this.cs_t and this.hs_t are based on w[], u[], and x_t[]
     * @post Resulting activations of this.fNetto, this.fNormalized, this.fScaledAndMoved, this.fActivated,
     *                                this.iNetto, this.iNormalized, this.iScaledAndMoved, this.iActivated,
     *                                this.cNetto, this.cNormalized, this.cScaledAndMoved, this.cActivated,
     *                                this.oNetto, this.oNormalized, this.oScaledAndMoved, this.oActivated are
     *                                based on w[], u[], and x_t[]
     * @return normalised pre-activations */
    void forwardPass(int timestep, float[]x_t) {                                                                        //wissenschaftliche Notation
        fNetto[timestep]= linearProjection(timestep, 'f', x_t);                                                    //zf
        iNetto[timestep]= linearProjection(timestep, 'i', x_t);                                                    //zi
        cNetto[timestep]= linearProjection(timestep, 'c', x_t);                                                    //zc
        oNetto[timestep]= linearProjection(timestep, 'o', x_t);                                                    //zo
        fNormalized[timestep]= preActivationNormalization(
                fNetto[timestep], fNetto[timestep][0],fNetto[timestep][1],iNetto[timestep][0],iNetto[timestep][1],
                                  cNetto[timestep][0],cNetto[timestep][1],oNetto[timestep][0],oNetto[timestep][1]);     //z'f
        iNormalized[timestep]= preActivationNormalization(
                iNetto[timestep], fNetto[timestep][0],fNetto[timestep][1],iNetto[timestep][0],iNetto[timestep][1],
                                  cNetto[timestep][0],cNetto[timestep][1],oNetto[timestep][0],oNetto[timestep][1]);     //z'i
        cNormalized[timestep]= preActivationNormalization(
                cNetto[timestep], fNetto[timestep][0],fNetto[timestep][1],iNetto[timestep][0],iNetto[timestep][1],
                                  cNetto[timestep][0],cNetto[timestep][1],oNetto[timestep][0],oNetto[timestep][1]);     //z'c
        oNormalized[timestep]= preActivationNormalization(
                oNetto[timestep], fNetto[timestep][0],fNetto[timestep][1],iNetto[timestep][0],iNetto[timestep][1],
                                  cNetto[timestep][0],cNetto[timestep][1],oNetto[timestep][0],oNetto[timestep][1]);     //z'o
        fScaledAndMoved[timestep]= scaleAndMove(fNormalized[timestep], fBeta, fGamma);                                  //yf
        iScaledAndMoved[timestep]= scaleAndMove(iNormalized[timestep], iBeta, iGamma);                                  //yi
        cScaledAndMoved[timestep]= scaleAndMove(cNormalized[timestep], cBeta, cGamma);                                  //yc
        oScaledAndMoved[timestep]= scaleAndMove(oNormalized[timestep], oBeta, oGamma);                                  //yo
        fActivated[timestep]= activation('f', fScaledAndMoved[timestep]);                                          //ft
        iActivated[timestep]= activation('i', iScaledAndMoved[timestep]);                                          //it
        cActivated[timestep]= activation('c', cScaledAndMoved[timestep]);                                          //ct
        oActivated[timestep]= activation('o', oScaledAndMoved[timestep]);                                          //ot
        cs_t[timestep]= calcNewCs_t(fActivated[timestep], (timestep==0)?cs_prev: cs_t[timestep-1], iActivated[timestep], cActivated[timestep]);
        hs_t[timestep]= calcNewHs_t(oActivated[timestep], cs_t[timestep]);
    }

    /**
     * @param timestep
     * @param yt the target
     * @param x current input
     * @post Resulting the states deltas in reference to target
     * @post Resulting the gates deltas in reference to target
     * @return resulting delL_delHSprev[] and delL_delCSprev[] */
    float[][]backwardPass(int timestep, LossFunc lossFunc, float[]yt, float[]x,
                          float[]f_normAct, float[]i_normAct, float[]c_normAct, float[]o_normAct) {
        float[]delL_delHS= lossFunc.calc(hs_t[timestep], yt);
        float[]delL_delCS= calcCsGradient(delL_delHS, o_normAct, cs_t[timestep]);

        // DONE: 15.09.2026 Aufspaltung auf die internen Vektoren
        float[]tanh= new float[cs_t[timestep].length];
        for(int i=0; i<cs_t[timestep].length; i++) {
            tanh[i]= (float)TANH.calc(cs_t[timestep][i]);
        }
        float[]sigmDeriv= new float[cs_t[timestep].length];
        for(int i=0; i<o_normAct.length; i++) {
            sigmDeriv[i]= o_normAct[i]* (1-o_normAct[i]);
        }
        // DONE: 25.09.2026 Post-Gate-Activation
        float[] delL_delO= chainrule(delL_delHS, tanh);
        float[] delL_delC= chainrule(delL_delCS, i_normAct);
        float[] delL_delI= chainrule(delL_delCS, c_normAct);
        float[] delL_delF= chainrule(delL_delCS, cs_prev);

        // DONE: 25.09.2026 Pre-Gate-Activation
        delL_delO= chainrule(delL_delO, sigmDeriv);
        delL_delC= chainrule(delL_delC, vecInvert(vecSquared(c_normAct)));
        delL_delI= chainrule(delL_delI, i_normAct, vecInvert(i_normAct));
        delL_delF= chainrule(delL_delF, f_normAct, vecInvert(f_normAct));

        // DONE: 15.09.2026 Berechne die partiellen Ableitungen der Gates und schreibe die Ergebnisse nach g[]
        final int gradientCardinality= delL_delF.length+ delL_delI.length+ delL_delC.length+ delL_delO.length;
        float[]g= new float[gradientCardinality];
        // DONE: 16.09.2026 Alle Werte/Ergebnisse der 4 partiellen Ableitungen nach g schreiben
        for(int i=0; i<delL_delF.length; i++) {
            g[i]= delL_delF[i];
        }
        for(int i=delL_delF.length; i<(delL_delF.length+delL_delI.length); i++) {
            g[i]= delL_delI[i-delL_delF.length];
        }
        for(int i=(delL_delF.length+delL_delI.length); i<(delL_delF.length+delL_delI.length+delL_delC.length); i++) {
            g[i]= delL_delC[i-(delL_delF.length+delL_delI.length)];
        }
        for(int i=(delL_delF.length+delL_delI.length+delL_delC.length); i<gradientCardinality; i++) {
            g[i]= delL_delO[i-(delL_delF.length+delL_delI.length+delL_delC.length)];
        }
        float[]preNormGradients= jacobianMatrix(delL_delF, delL_delI, delL_delC, delL_delO, g,
                                                fScaledAndMoved[timestep], iScaledAndMoved[timestep],
                                                cScaledAndMoved[timestep], oScaledAndMoved[timestep]);
        // DONE: 16.09.2026 Berechnung räumlicher Matrix-Gradienten via äußeres Produkt
        delL_delWf[timestep]= dyadicProduct(Arrays.copyOfRange(preNormGradients, 0, 2), x);
        delL_delWi[timestep]= dyadicProduct(Arrays.copyOfRange(preNormGradients, 2, 4), x);
        delL_delWc[timestep]= dyadicProduct(Arrays.copyOfRange(preNormGradients, 4, 6), x);
        delL_delWo[timestep]= dyadicProduct(Arrays.copyOfRange(preNormGradients, 6, 8), x);
        /*IMPORTANT: Nachtraeglich werden die f-/i-/c-/o-GewichtMatrizen per Delta-Regel geupdatet
                     wF(neu)= wF- ETA*delL_delWf
                     wI(neu)= wI- ETA*delL_delWi
                     wC(neu)= wC- ETA*delL_delWc
                     wO(neu)= wO- ETA*delL_delWo
         */

        delL_delUf[timestep]= dyadicProduct(Arrays.copyOfRange(preNormGradients, 0, 2), hs_prev);     //TODO: 20.09.2026 hs_prev wegoptimieren:= hs_t[timestep-1]
        delL_delUi[timestep]= dyadicProduct(Arrays.copyOfRange(preNormGradients, 2, 4), hs_prev);     //TODO: 20.09.2026 hs_prev wegoptimieren:= hs_t[timestep-1]
        delL_delUc[timestep]= dyadicProduct(Arrays.copyOfRange(preNormGradients, 4, 6), hs_prev);     //TODO: 20.09.2026 hs_prev wegoptimieren:= hs_t[timestep-1]
        delL_delUo[timestep]= dyadicProduct(Arrays.copyOfRange(preNormGradients, 6, 8), hs_prev);     //TODO: 20.09.2026 hs_prev wegoptimieren:= hs_t[timestep-1]
        /*IMPORTANT: Nachtraeglich werden die f-/i-/c-/o-rekurrenten GewichtMatrizen per Delta-Regel geupdatet
                     uF(neu)= uF- ETA*delL_delUf
                     uI(neu)= uI- ETA*delL_delUi
                     uC(neu)= uC- ETA*delL_delUc
                     uO(neu)= uO- ETA*delL_delUo
         */

        // DONE: 16.09.2026 Fehlerweiterleitung durch die Zeit
        // DONE: 24.09.2026 natürlich müssen die rekurrenten Matrizen(uf,ui,uc,uo) transponiert verrechnet werden
        float[]delL_delHSprev=  matAdd( matMul(transpose(uf), Arrays.copyOfRange(preNormGradients, 0, 2)),
                                        matMul(transpose(ui), Arrays.copyOfRange(preNormGradients, 2, 4)),
                                        matMul(transpose(uc), Arrays.copyOfRange(preNormGradients, 4, 6)),
                                        matMul(transpose(uo), Arrays.copyOfRange(preNormGradients, 6, 8)));
        /* DONE: 25.09.2026
            delL_delHSprev[] ist richtig
            delL_delCSprev[] ist richtig */
        float[]delL_delCSprev= hadamardProd(delL_delCS, f_normAct);
        float[][]result= new float[][]{delL_delHSprev, delL_delCSprev};
        return result;
    }

    private float[][] transpose(float[][] a) {
        float[][]result= new float[a[0].length][a.length];
        for(int c=0; c<a[0].length; c++) {
            for(int l=0; l<a.length; l++) {
                result[l][c]= a[c][l];
            }
        }
        return result;
    }

    /**
     * sums up given matrices
     * @param matSummands
     * @return summed matrix*/
    private float[] matAdd(float[]...matSummands) {
        float[] result= new float[matSummands[0].length];
        for(int i=0; i<matSummands.length; i++) {
            for(int r=0; r< result.length; r++) {
                result[r]+= matSummands[i][r];
            }
        }
        return result;
    }

    /**
     * dyadisches Produkt(Äußeres Produkt) Zwei Vektoren werden zu einer Matrix multipliziert.
     * @param col
     * @param line */
    private float[][] dyadicProduct(float[] col, float[]line) {
        float[][]res= new float[col.length][line.length];
        for(int c=0; c<col.length; c++) {
            for(int l=0; l<line.length; l++) {
                res[c][l]= col[c]* line[l];
            }
        }
        return res;
    }

    float[] chainrule(float[]...vecs) {
        float[] result= vecs[0];
        for(int i=1; i< vecs.length; i++) {
            result= hadamardProd(result, vecs[i]);
        }
        return result;
    }

    float[]vecInvert(float[]vec) {
        float[]result= new float[vec.length];
        for(int i=0; i<vec.length; i++) {
            result[i]= 1-vec[i];
        }
        return result;
    }

    float[]vecSquared(float[]vec) {
        float[]result= new float[vec.length];
        for(int i=0; i<vec.length; i++) {
            result[i]= (float)Math.pow(vec[i], 2);
        }
        return result;
    }

    private float[]calcCsGradient(float[] delL_delHS_prev, float[] oActivated_prev, float[] csT_prev) {
        float[]tanhDerivatCs= new float[csT_prev.length];
        for(int i=0; i<csT_prev.length; i++) {
            tanhDerivatCs[i]= (float)(1-Math.pow(TANH.calc(csT_prev[i]),2));
        }
        return hadamardProd(hadamardProd(delL_delHS_prev, oActivated_prev), tanhDerivatCs);
    }

    /**
     * @param gate
     * @param x_t
     * @return the combined linear projection     */
    float[]linearProjection(int timestep, char gate, float[]x_t) {
        switch(gate) {
            case 'f':  return vecAdd(matMul(wf, x_t), matMul(uf, (timestep==0)?hs_prev: hs_t[timestep-1]));
            case 'i':  return vecAdd(matMul(wi, x_t), matMul(ui, (timestep==0)?hs_prev: hs_t[timestep-1]));
            case 'c':  return vecAdd(matMul(wc, x_t), matMul(uc, (timestep==0)?hs_prev: hs_t[timestep-1]));
            default:   return vecAdd(matMul(wo, x_t), matMul(uo, (timestep==0)?hs_prev: hs_t[timestep-1]));
        }
    }

    /*Hadamard-Produkt (Elementweises Produkt)
      Zwei Matrizen[nXm] derselben Größe werden Position für Position miteinander multipliziert.
      Dabei werden zwei Vektoren oder Matrizen Zeile für Zeile multipliziert,
      aber das Ergebnis ist keine einzelne Zahl, sondern wieder ein neuer Vektor. */
    private float[]hadamardProd(float[]vecA, float[]vecB) {
        float[]res= new float[vecA.length];
        for(int i=0; i<res.length; i++) {
            res[i]= vecA[i]*vecB[i];
        }
        return res;
    }

    float[]calcNewCs_t(float[]f_t, float[]cs_prev, float[]i_t, float[]c_t) {
        return vecAdd(hadamardProd(f_t, cs_prev), hadamardProd(i_t, c_t));
    }

    float[]calcNewHs_t(float[]o_t, float[]cs_t) {
        float[]tanhOfCs_t= new float[cs_t.length];
        for(int i=0; i<cs_t.length; i++) {
            tanhOfCs_t[i]= (float)TANH.calc(cs_t[i]);
        }
        return hadamardProd(o_t, tanhOfCs_t);
    }

    private float[]matMul(float[][]mat, float[]vec) throws IllegalArgumentException{
        if(mat[0].length!= vec.length) throw new IllegalArgumentException();
        float[]result= new float[vec.length];
        for(int c=0; c< mat.length; c++) {
            for(int l=0; l< mat[0].length; l++) {
                result[c]+= mat[c][l]*vec[l];
            }
        }
        return result;
    }

    private float[]vecAdd(float[] vecA, float[] vecB) {
        float[]res= new float[vecB.length];
        for(int i=0; i<res.length; i++) {
            if(vecA!=null) {
                res[i] = vecA[i] + vecB[i];
            }else {
                res[i] = vecB[i];
            }
        }
        return res;
    }

    /**
     * normalizes vector a[]
     * @param a                     pair to normalize
     * @param aTotal                (joint) Layer Normalization über alle 4 Gates
     *                              Bei der Joint Layer Normalization (gemeinsame Layer-Normalisierung) werden
     *                              die Pre-Activations aller vier Gates (f, i, c, o) zu einem einzigen großen Vektor
     *                              zusammengefasst und gemeinsam über alle Dimensionen/Merkmale hinweg normalisiert.
     * @return  */
    float[]preActivationNormalization(float[]a, float...aTotal) {
        float[]a_hat= new float[a.length];
        double variance= 0;
        float sum= 0;
        for(float f:aTotal) sum+=f;
        float mean= sum/aTotal.length;
        for(float f:aTotal) variance+= Math.pow((f-mean), 2);
        variance/= aTotal.length;
        standardabweichung= Math.sqrt(variance+EPSILON);
        for(int i=0; i<a_hat.length; i++) {
            a_hat[i]= (float) ((a[i]-mean)/standardabweichung);
        }
        return a_hat;
    }

    /**
     * @param a
     * @param beta
     * @param gamma
     * @return  */
    float[]scaleAndMove(float[]a, float beta, float gamma) {
        float[]y= new float[a.length];
        for(int i=0; i<y.length; i++) {
            y[i]= a[i];
            y[i]*= gamma;
            y[i]+= beta;
        }
        return y;
    }

    /**
     * calls the corresponding activation function with the given parameter
     * @param gate  to determine the relevant activation func
     * @param a
     * @return  */
    float[]activation(char gate, float[] a) {
        float[]result= new float[a.length];
        switch(gate) {
            case 'c':  for(int i=0; i<result.length; i++) {result[i]= (float)TANH.calc(a[i]);}break;
            default:   for(int i=0; i<result.length; i++) {result[i]= (float)SIGMOID.calc(a[i]);}break;
        }
        return result;
    }

    /**
     *
     * @param delLdelf  [interne Vektoren nach aufspaltung in BPTT]
     *                  Gradient um Vektor auf Zustand nach seiner Normalisierung[Forwardpass] zeigen zu lassen.
     *                  := g...
     * @param delLdeli  [interne Vektoren nach aufspaltung in BPTT]
     *                  Gradient um Vektor auf Zustand nach seiner Normalisierung[Forwardpass] zeigen zu lassen.
     *                  := ...g...
     * @param delLdelc  [interne Vektoren nach aufspaltung in BPTT]
     *                  Gradient um Vektor auf Zustand nach seiner Normalisierung[Forwardpass] zeigen zu lassen.
     *                  := ...g...
     * @param delLdelo  [interne Vektoren nach aufspaltung in BPTT]
     *                  Gradient um Vektor auf Zustand nach seiner Normalisierung[Forwardpass] zeigen zu lassen.
     *                  := ...g
     * @param g :=Der vollständige Gradientenvektor
     * @param f_scaledMoved :=a^
     * @param i_scaledMoved :=a^
     * @param c_scaledMoved :=a^
     * @param o_scaledMoved :=a^
     * @return     */
    float[]jacobianMatrix(float[]delLdelf, float[]delLdeli, float[]delLdelc, float[]delLdelo, float[]g,
                          float[]f_scaledMoved, float[]i_scaledMoved, float[]c_scaledMoved, float[]o_scaledMoved) {
        // DONE: 23.09.2026 Das muss jetzt als Ablauf generalisiert werden
        float[] res= new float[delLdelf.length+ delLdeli.length+ delLdelc.length+ delLdelo.length];
        float sumOfG= 0;
        for(float f:g) sumOfG+= f;
        float Mittelwert= sumOfG/g.length;
        float skalarprodukt= 0;
        int g_elm= 0;

        for(int i=0; i< f_scaledMoved.length; i++)
            skalarprodukt+= g[g_elm++]* f_scaledMoved[i];
        for(int i=0; i< i_scaledMoved.length; i++)
            skalarprodukt+= g[g_elm++]* i_scaledMoved[i];
        for(int i=0; i< c_scaledMoved.length; i++)
            skalarprodukt+= g[g_elm++]* c_scaledMoved[i];
        for(int i=0; i< o_scaledMoved.length; i++)
            skalarprodukt+= g[g_elm++]* o_scaledMoved[i];

        float[]fJacobiProd= new float[f_scaledMoved.length];
        float[]iJacobiProd= new float[i_scaledMoved.length];
        float[]cJacobiProd= new float[c_scaledMoved.length];
        float[]oJacobiProd= new float[o_scaledMoved.length];
        for(int i=0; i< cs_t.length; i++) {
            fJacobiProd[i] = (1f / (float) standardabweichung) * (delLdelf[i] - Mittelwert - (1f / 8f) * skalarprodukt * f_scaledMoved[i]);
            iJacobiProd[i] = (1f / (float) standardabweichung) * (delLdeli[i] - Mittelwert - (1f / 8f) * skalarprodukt * i_scaledMoved[i]);
            cJacobiProd[i] = (1f / (float) standardabweichung) * (delLdelc[i] - Mittelwert - (1f / 8f) * skalarprodukt * c_scaledMoved[i]);
            oJacobiProd[i] = (1f / (float) standardabweichung) * (delLdelo[i] - Mittelwert - (1f / 8f) * skalarprodukt * o_scaledMoved[i]);
        }
        System.arraycopy(fJacobiProd, 0, res, 0, fJacobiProd.length);
        System.arraycopy(iJacobiProd, 0, res, 2, iJacobiProd.length);
        System.arraycopy(cJacobiProd, 0, res, 4, cJacobiProd.length);
        System.arraycopy(oJacobiProd, 0, res, 6, oJacobiProd.length);
        return res;
    }
}
