package lstm2604221130;

import org.junit.jupiter.api.Test;

import static java.lang.Math.log;
import static org.junit.jupiter.api.Assertions.*;

public class Test_DenseLayer {

    @Test
    void testDenseLayer_ForwardingBackpropagationMitAktivierungsfunktionMitBiasF() {
        /*  Google: "LSTM was macht ein dense-layer? Gib mir ein rechenbeispiel für forwardpass und backpropagation mit
                     input={0.5, -0.2, 0.1, 0.8} und Gewichten={{0.2, 0.3}, {0.5, -0.3}, {-0.1, 0.6}, {0.4, 0.1}},
                     bias={0.1, 0.2} und fehlerGradient={0.4, 1.2}."
         */
        float DELTA= 1e-3f;
        float[] hT= new float[]{0.5f, -0.2f, 0.1f, 0.8f};                             //input[als Zeilenvektor]
        float[][] w= new float[][]{{0.2f, 0.3f}, {0.5f, -0.3f}, {-0.1f, 0.6f}, {0.4f, 0.1f}};   //gewichte[als Spaltenvektor]
        float[] b= new float[]{0.1f, 0.2f};                                     //bias

        DenseLayer denseLayer= new DenseLayer(hT, w, b);
        assertArrayEquals(denseLayer.getOutput(), new float[]{0.6011f, 0.6341f}, DELTA);   //:={sigmoid(-0.9),sigmoid(1.2)}
        float[] fehlerGradient=new float[]{0.4f, 1.2f};
        float[][]backpropagatededGradients= denseLayer.backpropagate(fehlerGradient, hT, w);
        assertArrayEquals(backpropagatededGradients[0], new float[]{0.2f, -0.08f, 0.04f, 0.32f}, DELTA);
        assertArrayEquals(backpropagatededGradients[1], new float[]{0.6f, -0.24f, 0.12f, 0.96f}, DELTA);

        assertArrayEquals(denseLayer.getWeights()[0], new float[]{0.18f, 0.24f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[1], new float[]{0.508f, -0.276f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[2], new float[]{-0.104f, 0.588f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[3], new float[]{0.368f, 0.004f}, DELTA);
        assertArrayEquals(denseLayer.getBiases(), new float[]{0.06f, 0.08f}, DELTA);
    }

    @Test
    void testDenseLayer_ForwardingBackpropagationMitAktivierungsfunktionMitBiasE() {
        /*  Google: "LSTM was macht ein dense-layer? Gib mir ein rechenbeispiel für forwardpass und backpropagation mit
                     input={1, 0, 2, -1, 0.5} und
                     Gewichten={{0.2, -0.3}, {0.5, 0.1}, {-0.1, 0.4}, {0.8, -0.2}, {-0.4, 0.6}}, bias={0.1, 0.2} und
                     fehlerGradient={1.0, 1.0}."
         */
        float DELTA= 1e-3f;
        float[] hT= new float[]{1f, 0f, 2f, -1f, 0.5f};                             //input[als Zeilenvektor]
        float[][] w= new float[][]{{0.2f, -0.3f}, {0.5f, 0.1f}, {-0.1f, 0.4f}, {0.8f, -0.2f}, {-0.4f, 0.6f}};   //gewichte[als Spaltenvektor]
        float[] b= new float[]{0.1f, 0.2f};                                     //bias

        DenseLayer denseLayer= new DenseLayer(hT, w, b);
        assertArrayEquals(denseLayer.getOutput(), new float[]{0.2891f, 0.7685f}, DELTA);   //:={sigmoid(-0.9),sigmoid(1.2)}
        float[] fehlerGradient=new float[]{1f, 1f};
        float[][]backpropagatededGradients= denseLayer.backpropagate(fehlerGradient, hT, w);
        assertArrayEquals(backpropagatededGradients[0], new float[]{1f, 0f, 2.0f, -1.0f, 0.5f}, DELTA);
        assertArrayEquals(backpropagatededGradients[1], new float[]{1f, 0f, 2.0f, -1.0f, 0.5f}, DELTA);

        assertArrayEquals(denseLayer.getWeights()[0], new float[]{0.1f, -0.4f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[1], new float[]{0.5f, 0.1f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[2], new float[]{-0.3f, 0.2f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[3], new float[]{0.9f, -0.1f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[4], new float[]{-0.45f, 0.55f}, DELTA);
        assertArrayEquals(denseLayer.getBiases(), new float[]{0f, 0.1f}, DELTA);
    }

    @Test
    void testDenseLayer_ForwardingBackpropagationMitAktivierungsfunktionMitBiasD() {
        /*  Google: "LSTM was macht ein dense-layer? Gib mir ein rechenbeispiel für forwardpass und backpropagation mit
                     input={5 dimensional}, und zufälligen Gewichten, bias={0.1, 0.2} und
                     fehlerGradient={1.0, 1.0}."
         */
        float DELTA= 1e-3f;
        float[] hT= new float[]{1f, 2f, 3f, 4f, 5f};                             //input[als Zeilenvektor]
        float[][] w= new float[][]{{0.5f, 0.3f}, {-0.2f, 0.7f}, {0.1f, -0.5f}, {0.8f, 0.2f}, {-0.4f, 0.6f}};   //gewichte[als Spaltenvektor]
        float[] b= new float[]{0.1f, 0.2f};                                     //bias

        DenseLayer denseLayer= new DenseLayer(hT, w, b);
        assertArrayEquals(denseLayer.getOutput(), new float[]{0.8455f, 0.9852f}, DELTA);   //:={sigmoid(1.7),sigmoid(4.2)}
        float[] fehlerGradient=new float[]{0.131f, 0.015f};
        float[][]backpropagatededGradients= denseLayer.backpropagate(fehlerGradient, hT, w);
        assertArrayEquals(backpropagatededGradients[0], new float[]{0.131f, 0.262f, 0.393f, 0.524f, 0.655f}, DELTA);
        assertArrayEquals(backpropagatededGradients[1], new float[]{0.015f, 0.030f, 0.045f, 0.060f, 0.075f}, DELTA);

        assertArrayEquals(denseLayer.getWeights()[0], new float[]{0.4869f, 0.2985f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[1], new float[]{-0.2262f, 0.6970f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[2], new float[]{0.0607f, -0.5045f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[3], new float[]{0.7476f, 0.1940f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[4], new float[]{-0.4655f, 0.5925f}, DELTA);
        assertArrayEquals(denseLayer.getBiases(), new float[]{0.0869f, 0.1985f}, DELTA);
        /*
            Was ist passiert?
            Da wir im Backpropagation-Schritt einen positiven Fehler-Gradienten (1.0, 1.0) angenommen haben
            (was bedeutet: "Der Output ist zu hoch"), hat der Optimizer die Gewichte und Biases verringert.
            Wie du siehst, sind beide Output-Werte im zweiten Durchgang deutlich gesunken.
         */
    }

    @Test
    void testDenseLayer_ForwardingBackpropagationMitAktivierungsfunktionMitBiasC() {
        /*  Google: "LSTM was macht ein dense-layer?
                     Gib mir ein Rechenbeispiel für forwardpass mit 4 dimensionalem input und backpropagation. "
         */
        float DELTA= 1e-3f;
        float[] hT= new float[]{1.0f, 0.f, 2f, -1f};                 //input[als Zeilenvektor]
        float[][] w= new float[][]{{0.5f}, {0.2f}, {-0.1f}, {0.8f}};      //gewichte[als Spaltenvektor bzw transponiert]
        float[] b= new float[]{0.1f};                                     //bias

        DenseLayer denseLayer= new DenseLayer(hT, w, b);
        assertArrayEquals(denseLayer.getOutput(), new float[]{0.4013f}, DELTA);   //:={sigmoid(-0.4)}
        float[] fehlerGradient=new float[]{-3f};
        assertArrayEquals(denseLayer.backpropagate(fehlerGradient, hT, w, true), new float[]{-3f, 0f, -6f, 3f}, DELTA);

        assertArrayEquals(denseLayer.getWeights()[0], new float[]{0.8f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[1], new float[]{0.2f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[2], new float[]{0.5f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[3], new float[]{0.5f}, DELTA);
        assertArrayEquals(denseLayer.getBiases(), new float[]{0.4f}, DELTA);
    }

    @Test
    void testDenseLayer_ForwardingBackpropagationMitAktivierungsfunktionMitBiasB() {
        /*  Google: "LSTM was macht ein dense-layer?
                     Gib mir ein Rechenbeispiel für forwardpass mit 4 dimensionalem input und backpropagation. "
         */
        float DELTA= 1e-3f;
        float[] hT= new float[]{1.0f, 0.5f, -0.2f, 0.8f};                 //input[als Zeilenvektor]
        float[][] w= new float[][]{{0.2f}, {-0.3f}, {0.5f}, {0.1f}};      //gewichte[als Spaltenvektor]
        float[] b= new float[]{0.1f};                                     //bias

        DenseLayer denseLayer= new DenseLayer(hT, w, b);
        assertArrayEquals(denseLayer.getOutput(), new float[]{0.5325f}, DELTA);   //:={sigmoid(0.13)}
        float[] fehlerGradient=new float[]{-0.37f};
        assertArrayEquals(denseLayer.backpropagate(fehlerGradient, hT, w, true), new float[]{-0.37f, -0.185f, 0.074f, -0.296f}, DELTA);

        assertArrayEquals(denseLayer.getWeights()[0], new float[]{0.237f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[1], new float[]{-0.2815f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[2], new float[]{0.4926f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[3], new float[]{0.1296f}, DELTA);
        assertArrayEquals(denseLayer.getBiases(), new float[]{0.137f}, DELTA);
    }

    @Test
    void testDenseLayer_ForwardingBackpropagationMitAktivierungsfunktionMitBiasA() {
        /*  Google: "LSTM was macht ein dense-layer rechenbeispiel für forwardpass und backpropagation mit
                     input={1.0, 0.5, -0.2}, Gewichte={{0.1, 0.4}, {0.2, 0.5}, {0.3, 0.6}}, bias={0.1, 0.2} und
                     fehlerGradient={1.0, 1.0}."
         */
        float DELTA= 1e-3f;
        float[] hT= new float[]{1.0f, 0.5f, -0.2f};                             //input[als Zeilenvektor]
        float[][] w= new float[][]{{0.1f, 0.4f}, {0.2f, 0.5f}, {0.3f, 0.6f}};   //gewichte[als Spaltenvektor]
        float[] b= new float[]{0.1f, 0.2f};                                     //bias

        DenseLayer denseLayer= new DenseLayer(hT, w, b);
        assertArrayEquals(denseLayer.getOutput(), new float[]{0.55971f, 0.6748f}, DELTA);   //:={sigmoid(0.24),sigmoid(0.73)}
        float[] fehlerGradient=new float[]{1f, 1f};
        assertArrayEquals(denseLayer.backpropagate(fehlerGradient, hT, w, false), new float[]{0.5f, 0.7f, 0.9f}, DELTA);

        assertArrayEquals(denseLayer.getWeights()[0], new float[]{0f, 0.3f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[1], new float[]{0.15f, 0.45f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[2], new float[]{0.32f, 0.62f}, DELTA);
        assertArrayEquals(denseLayer.getBiases(), new float[]{0f, 0.1f}, DELTA);

        denseLayer= new DenseLayer(hT, w, b);
        assertArrayEquals(denseLayer.getOutput(), new float[]{0.50275f, 0.6227f}, DELTA);   //:={sigmoid(0.011),sigmoid(0.501)}
        /*
            Was ist passiert?
            Da wir im Backpropagation-Schritt einen positiven Fehler-Gradienten (1.0, 1.0) angenommen haben
            (was bedeutet: "Der Output ist zu hoch"), hat der Optimizer die Gewichte und Biases verringert.
            Wie du siehst, sind beide Output-Werte im zweiten Durchgang deutlich gesunken.
         */
    }

    @Test
    void testDenseLayer_ForwardingBackpropagationOhneAktivierungsfunktionMitBiasA() {
        //1. Forward Propagation
        float DELTA= 1e-3f;
        float[] hT= new float[]{1.0f, 0.5f, -0.2f};                             //input[als Zeilenvektor]
        float[][] w= new float[][]{{0.1f, 0.4f}, {0.2f, 0.5f}, {0.3f, 0.6f}};   //gewichte[als Spaltenvektor]
        float[] b= new float[]{0.1f, 0.2f};                                     //bias

        DenseLayer denseLayer= new DenseLayer(hT, w, b);
        float y=1f;
        denseLayer.backpropagate(y, hT, w);

        assertArrayEquals(denseLayer.getWeights()[0], new float[]{0.11085016f, 0.4054251f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[1], new float[]{0.21085016f, 0.5054251f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[2], new float[]{0.31085016f, 0.6054251f}, DELTA);
    }

    @Test
    void testDenseLayer_ForwardingBackpropagationB() {
        /*https://gemini.google.com
            "Rechenbeispiel Dense-Layer aus LSTM für Forwarding und Backpropagation,
             verwende als Bias b=-0.1, als input x = {0.5f, -0.2}, als Gewichte w={{0.4},{0.8}}
             und als Target y=1.0. Verwende Aktivierungsfunktion Sigmoid für die Ausgabe der Forward Propagation."
         */
      //1. Forward Propagation
        float DELTA= 1e-3f;
        float[] hT= new float[]{0.5f, -0.2f};           //input[als Zeilenvektor]
        float[][] w= new float[][]{{0.4f},{0.8f}};      //gewichte[als Spaltenvektor]
        float[] b= new float[]{-0.1f};                  //bias

        DenseLayer denseLayer= new DenseLayer(hT, w, b);
        assertArrayEquals(denseLayer.getOutput(), new float[]{0.485f}, DELTA);
      //2. Loss Function            Nehmen wir an, das Target/echte Label ist y=1. Wir nutzen den Binary Cross-Entropy Loss:
        float y=1f;
        float L=(float) ( (0.5)*Math.pow((denseLayer.getOutput()[0]-1),2) );      //L=[MSE]
      //3. Backpropagation (Der "Dense" Teil)
       //Schritt A: Ableitungen berechnen
        //1. Loss nach Aktivierung
        float delLdelYhut= denseLayer.getOutput()[0]- y;
        assertEquals(-0.515, delLdelYhut, DELTA);
        //2. Aktivierung nach gewichteter Input(SIGMOID_DERIVATIVE)
        float aktivierungNachZ= denseLayer.getOutput()[0]*(1-denseLayer.getOutput()[0]);
        assertEquals(0.2498, aktivierungNachZ, DELTA);

        //2. Gradient der Gewichte (W)
        float fehlerSignal= delLdelYhut* aktivierungNachZ;                  //"Lokaler Gradient"
        assertEquals(-0.1286, fehlerSignal, DELTA);
        float[] delLdelW= new float[]{fehlerSignal * hT[0], fehlerSignal * hT[1]};      //jedes Element aus hT[] wird mit Gradient des Loss multipliziert
        assertArrayEquals(delLdelW, new float[]{-0.0643f, 0.0257f}, DELTA);

        //3. Gradient des Bias (b)
        float delLdelB= fehlerSignal;
        assertEquals(delLdelB, -0.1286f, DELTA);

      //4. Backpropagation update Weights/Biases
        denseLayer.backpropagate(y, hT, w);
        assertArrayEquals(denseLayer.getWeights()[0], new float[]{0.40643167f}, DELTA);
        assertArrayEquals(denseLayer.getWeights()[1], new float[]{0.80643165f}, DELTA);
        assertArrayEquals(denseLayer.getBiases(), new float[]{-0.10128633f}, DELTA);

      //5. Backpropagation in das LSTM hinein
    }

    @Test
    void testDenseLayer_ForwardingBackpropagationA() {
      //1. Forward Propagation
        float DELTA= 1e-3f;
        float[] hT= new float[]{0.5f, -0.2f};           //input[als Zeilenvektor]
        float[][] w= new float[][]{{0.4f},{0.7f}};      //gewichte[als Spaltenvektor]
        float[] b= new float[]{0.1f};                   //bias

        DenseLayer denseLayer= new DenseLayer(hT, w, b);
        assertArrayEquals(denseLayer.getOutput(), new float[]{0.54f}, DELTA);
      //2. Loss Function            Nehmen wir an, das echte Label ist y=1. Wir nutzen den Binary Cross-Entropy Loss:
        float y=1f;
        float L=(-1)*(float)log(0.54);      //L=−(y log(y^hut)+(1−y)log(1−y^hut))
      //3. Backpropagation (Der "Dense" Teil)
       //1. Gradient des Loss nach der Aktivierung  ∂L/∂y^hut= y^hut−y= 0.54−1= −0.46
        float delLdelYhut= denseLayer.getOutput()[0]- y;

       //2. Gradient der Gewichte (W)
        float[] delLdelW= new float[]{delLdelYhut * hT[0], delLdelYhut * hT[1]};      //jedes Element aus hT[] wird mit Gradient des Loss multipliziert
        assertArrayEquals(delLdelW, new float[]{-0.23f, 0.092f}, DELTA);

       //3. Gradient des Bias (b)
        float delLdelB= delLdelYhut* 1;
        assertEquals(delLdelB, -0.46f, DELTA);

      //4. Backpropagation in das LSTM hinein           Wir berechnen den Gradienten bezüglich des Inputs hT[]
        float[] delLdelhT= new float[]{delLdelYhut* w[0][0], delLdelYhut* w[1][0]};
        assertArrayEquals(delLdelhT, new float[]{-0.184f, -0.322f}, DELTA);

        /*
        Dieser Vektor <delLdelYhut> ist nun der "Fehler-Input" für das LSTM.
        Von hier aus startet BPTT (Backpropagation Through Time),
        um die internen Gates (Forget, Input, Output Gate) des LSTM anzupassen.
         */

      //Update Weights and Bias(es)
        float LEARNING_RATE= 0.1f;
        w[0][0]-=(LEARNING_RATE* delLdelW[0]);
        assertEquals(w[0][0], 0.423f, DELTA);
        w[1][0]-=(LEARNING_RATE* delLdelW[1]);

        b[0]-=(LEARNING_RATE* delLdelB);
        assertEquals(b[0], 0.146f, DELTA);

    }

    @Test
    void testDenseLayerForwarding() {
        float DELTA= 1e-3f;
        float[]input= new float[]{1.0f, 0.5f};
        float[][]weightMatrix= new float[][]   {
                {0.2f, 0.8f},
                {0.5f, -0.1f}
        };
        float[]bias= new float[]{0.1f, 0.2f};
        DenseLayer denseLayer= new DenseLayer(input, weightMatrix, bias);

//        assertTrue(denseLayer.getOutput()[0]==0.55f);   //Ohne Aktivierungsfunktion[sigmoid]
//        assertTrue(denseLayer.getOutput()[1]==0.95f);   //Ohne Aktivierungsfunktion[sigmoid]

//        assertArrayEquals(denseLayer.getOutput(), new float[]{0.55f, 0.95f}, DELTA);    //Ohne Aktivierungsfunktion[sigmoid]
//        assertArrayEquals(denseLayer.getOutput(), new float[]{0.61f, 0.679f}, DELTA);   //Mit Aktivierungsfunktion[sigmoid]
        assertArrayEquals(denseLayer.getOutput(), new float[]{0.634f, 0.721f}, DELTA);   //Mit Aktivierungsfunktion[sigmoid]
    }

    @Test
    void testDenseLayer() {
        float DELTA= 1e-3f;
        float[][]weights= new float[][]{{0.2f, 0.8f},
                {0.5f, -0.1f},
                {0.1f, 0.3f}};
        DenseLayer denseLayer= new DenseLayer(new float[]{1.0f, 0.5f, -0.2f},
                weights, new float[]{0.1f, 0.2f, 0.3f});
        assertArrayEquals( denseLayer.getOutput() ,
//                new float[]{0.43f+0.1f, 0.69f+0.2f} , DELTA);     //Mit Bias; Ohne Aktivierungsfunktion[sigmoid]
//                new float[]{0.605f,0.666f} , DELTA);            //Ohne Bias; Mit Aktivierungsfunktion[sigmoid]
                new float[]{0.63f,0.709f} , DELTA);            //Mit Bias; Mit Aktivierungsfunktion[sigmoid]
    }
    @Test
    void testDenseLayerOverloadConstructor() {
        DenseLayer denseLayer= new DenseLayer(new float[]{1.0f, 0.5f, -0.2f}, 3);
        assertTrue(true);
    }

    @Test
    void anotherDenseLayerTest() {
        float DELTA= 1e-3f;
        float[]input= new float[]{1.0f, 0.5f, -0.2f};
        float[][]weightMatrix= new float[][]   {{0.2f, 0.8f},
                                                {0.5f, -0.1f},
                                                {0.1f, 0.3f}};
        float[]bias= new float[]{0.1f, 0.2f};
        DenseLayer denseLayer= new DenseLayer(input, weightMatrix, bias);

        assertArrayEquals( denseLayer.getOutput() , new float[]{0.629f, 0.709f}, DELTA);   //Mit Aktivierungsfunktion[sigmoid], mit Bias

//        assertTrue(denseLayer.getOutput()[0]==0.53f);   //Mit Bias ohne Aktivierungsfunktion[sigmoid]
//        assertTrue(denseLayer.getOutput()[1]==0.89f);   //Mit Bias ohne Aktivierungsfunktion[sigmoid]
    }
}
