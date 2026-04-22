package lstm2604221130;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class Test_BatchNormalizationLayer {
    final BatchNormalizationLayer batchNormalizationLayer= new BatchNormalizationLayer();
    float BETA= 0.f;
    float DELTA= 1e-3f;
    float GAMMA= 1f;
    float EPSILON= 1e-5f;   //für nummerische Stabilität

    @Test
    void testBatchNormalizationForwardingBackpropagationF() {
        float[] miniBatch= new float[]{10f, 20f, 30f};
        batchNormalizationLayer.forward(EPSILON, miniBatch);
        assertArrayEquals(batchNormalizationLayer.getAverage(miniBatch), new float[]{20f}, DELTA);
        assertArrayEquals(batchNormalizationLayer.getVariance(  new float[][]{miniBatch},
                        batchNormalizationLayer.getAverage(miniBatch)),
                new float[]{66.666666f},
                DELTA);
        float[][] normed= batchNormalizationLayer.getNormalization( new float[][]{miniBatch},
                batchNormalizationLayer.getAverage(miniBatch),
                batchNormalizationLayer.getVariance(new float[][]{miniBatch},
                        batchNormalizationLayer.getAverage(miniBatch)),
                EPSILON);
        assertArrayEquals(normed[0], new float[]{-1.225f, 0, 1.225f}, DELTA);

        //Angenommen, die zurückgeführten Gradienten vom nächsten Layer sind
        float[][] dLdy= new float[][]{{2f, 1f, -3f}};
        assertArrayEquals(
                batchNormalizationLayer.backpropagate(
                        miniBatch.length, miniBatch, BETA, GAMMA, EPSILON, dLdy[0]
                )[0],
                new float[]{-0.0613f,0.1224f, -0.0611f},
                DELTA
        );

        /*
            Die neuen Werte für den nächsten Durchlauf[für eta/Lernrate= 0.1]:
                x1[neu]: 10-0.1*(-0.0613) = 10.00613
                x2[neu]: 20-0.1* (0.1224) = 19.98776
                x3[neu]: 30-0.1* (0.0611) = 30.00611
            Damit resultiert daraus eine optimierte miniBatch von {10.00613, 19.98776, 30.00611}
         */
    }

    @Test
    void testBatchNormalizationForwardingBackpropagationE() {
        float[] miniBatch= new float[]{12.34f, 23.45f, 34.56f};
        batchNormalizationLayer.forward(EPSILON, miniBatch);
        assertArrayEquals(batchNormalizationLayer.getAverage(miniBatch), new float[]{23.45f}, DELTA);
        assertArrayEquals(batchNormalizationLayer.getVariance(  new float[][]{miniBatch},
                        batchNormalizationLayer.getAverage(miniBatch)),
                new float[]{82.2887f},
                DELTA);
        float[][] normed= batchNormalizationLayer.getNormalization( new float[][]{miniBatch},
                batchNormalizationLayer.getAverage(miniBatch),
                batchNormalizationLayer.getVariance(new float[][]{miniBatch},
                        batchNormalizationLayer.getAverage(miniBatch)),
                EPSILON);
        assertArrayEquals(normed[0], new float[]{-1.2247f, 0, 1.2247f}, DELTA);

        //Angenommen, die zurückgeführten Gradienten vom nächsten Layer sind
        float[][] dLdy= new float[][]{{90.78f, 87.67f, 76.56f}};
        assertArrayEquals(
                batchNormalizationLayer.backpropagate(
                        miniBatch.length, miniBatch, BETA, GAMMA, EPSILON, dLdy[0]
                )[0],
                new float[]{-0.1469f,0.2940f, -0.1471f},
                DELTA
        );
    }

    @Test
    void testBatchNormalizationForwardingBackpropagationD() {
        float[] miniBatch= new float[]{0.9f,8.7f,6.5f,4.3f,2.1f};
        batchNormalizationLayer.forward(EPSILON, miniBatch);
        assertArrayEquals(batchNormalizationLayer.getAverage(miniBatch), new float[]{4.5f}, DELTA);
        assertArrayEquals(batchNormalizationLayer.getVariance(  new float[][]{miniBatch},
                        batchNormalizationLayer.getAverage(miniBatch)),
                new float[]{8.08f},
                DELTA);
        float[][] normed= batchNormalizationLayer.getNormalization( new float[][]{miniBatch},
                batchNormalizationLayer.getAverage(miniBatch),
                batchNormalizationLayer.getVariance(new float[][]{miniBatch},
                        batchNormalizationLayer.getAverage(miniBatch)),
                EPSILON);
        assertArrayEquals(normed[0], new float[]{-1.266f, 1.477f, 0.704f, -0.07f, -0.844f}, DELTA);

        //Angenommen, die zurückgeführten Gradienten vom nächsten Layer sind
        float[][] dLdy= new float[][]{{4.2f, 42f, 0.42f, 0.042f, 4.2f}};
        assertArrayEquals(
                batchNormalizationLayer.backpropagate(
                        miniBatch.length, miniBatch, BETA, GAMMA, EPSILON, dLdy[0]
                )[0],
                new float[]{2.665f,5.637f,-6.079f,-3.3f,1.076f},
                DELTA
        );
    }

    @Test
    void testBatchNormalizationForwardingBackpropagationC() {                                                           //https://www.google.com/search?q=lstm+batch+normalization+backpropagation+layer+numerisches+rechenbeispiel+mit+mini-batch+m%3D3&client=firefox-b-d&hs=6PvU&sca_esv=3eb9bc99a534cb7d&fbs=ADc_l-YGrpJMQtvjQ6h14rj-dfIrbPkd_Upq68wJVnEIgo2Pwxu679PACFfTKv4n_1_FsyWRtH6VXdFUsrvx5e4Eax_Xgq10hasdmEAWZSPWsvmYCxkgxhgAAEJUbue-qZnt86OK4vP6UejxniOK-dq6UFTtv5LiMcQFF9beBU00mOoHg8GHTk3CzOojQQPhHrd-1Qm9JKHi&aep=1&ntc=1&sa=X&ved=2ahUKEwjn8qjLv72TAxUKgP0HHTK-HsgQ2J8OegQIDxAE&biw=1486&bih=731&dpr=1.25&mstk=AUtExfDggd6fQRXQAqz_Q7wPe-RZkBXEo6aimVzpEQZeKYTYLw7-GxFV8fZ9AZIsE4TJ0RoiKKp3Xdo8ultV6aHGHLupDgrrHWf9PF4ZxAUgHOJ6EeNlPjKqBVZr9dPCm51I7SWPLAPpef2DPW9ct-4G6_UYWoQQugYpZPU&csuir=1&sei=0WrKaZr2N8eE9u8PmJrJuQc&udm=50
        float[] miniBatch= new float[]{1.2f,3.4f,5.6f};
        batchNormalizationLayer.forward(EPSILON, miniBatch);
        assertArrayEquals(batchNormalizationLayer.getAverage(miniBatch), new float[]{3.4f}, DELTA);
        assertArrayEquals(batchNormalizationLayer.getVariance(  new float[][]{miniBatch},
                        batchNormalizationLayer.getAverage(miniBatch)),
                new float[]{3.2267f},
                DELTA);
        float[][] normed= batchNormalizationLayer.getNormalization( new float[][]{miniBatch},
                batchNormalizationLayer.getAverage(miniBatch),
                batchNormalizationLayer.getVariance(new float[][]{miniBatch},
                        batchNormalizationLayer.getAverage(miniBatch)),
                EPSILON);
        assertArrayEquals(normed[0], new float[]{-1.225f, 0f, 1.225f}, DELTA);

        //Angenommen, die zurückgeführten Gradienten vom nächsten Layer sind
        float[][] dLdy= new float[][]{{1.0f, 5.0f, 2.0f}};
        assertArrayEquals(
                batchNormalizationLayer.backpropagate(
                        miniBatch.length, miniBatch, BETA, GAMMA, EPSILON, dLdy[0]
                )[0],
                new float[]{-0.6496f,1.2992f,-0.6496f},
                DELTA
        );
    }

    @Test
    void testBatchNormalizationForwardingBackpropagationMitZweiFeaturesC() {
        float[][] miniBatches= new float[][]{{1.2f, 3.4f, 5.6f}, {6.5f, 4.3f, 2.1f}};                                   //Das entspr. eigentlich m=3:= {{1.2f,6.5f},{3.4f,4.3f},{5.6f,2.1f}}
        batchNormalizationLayer.forward(EPSILON, miniBatches);
        assertArrayEquals(batchNormalizationLayer.getAverage(miniBatches), new float[]{3.4f,4.3f}, DELTA);
        assertArrayEquals(
                batchNormalizationLayer.getVariance(new float[][]{miniBatches[0]},
                        batchNormalizationLayer.getAverage(miniBatches[0])
                ),
                new float[]{3.226f},
                DELTA);
        assertArrayEquals(
                batchNormalizationLayer.getVariance(new float[][]{miniBatches[1]},
                        batchNormalizationLayer.getAverage(miniBatches[1])
                ),
                new float[]{3.226f},
                DELTA);
        float[][] normed= batchNormalizationLayer.getNormalization( miniBatches,
                batchNormalizationLayer.getAverage(miniBatches),
                batchNormalizationLayer.getVariance(miniBatches,
                        batchNormalizationLayer.getAverage(miniBatches)),
                EPSILON);
        assertArrayEquals(normed[0], new float[]{-1.225f, 0f, 1.225f}, DELTA);                                          //Die normalisierten Werte des 1. Features
        assertArrayEquals(normed[1], new float[]{1.225f, 0f, -1.225f}, DELTA);                                          //Die normalisierten Werte des 2. Features

        float[][] dy= new float[][]{{-1f, 0.42f, -2f}, {2f, 0.2f, 0.42f}};                                              //Eingehender Gradient vom Loss (dy)

        assertArrayEquals(
                batchNormalizationLayer.backpropagate(
                        miniBatches[0].length, miniBatches[0], BETA, GAMMA, EPSILON, dy[0]
                )[0],
                new float[]{-0.356f,0.713f,-0.356f},
                DELTA
        );
        assertArrayEquals(
                batchNormalizationLayer.backpropagate(
                        miniBatches[1].length, miniBatches[1], BETA, GAMMA, EPSILON, dy[1]
                )[0],
                new float[]{0.187f,-0.375f,0.187f},
                DELTA
        );
    }

    @Test
    void testBatchNormalizationForwardingBackpropagationMitZweiFeaturesB() {
        float[][] miniBatches= new float[][]{{30f, 20f, 10f}, {2f, 4f, 6f}};     //Das entspr. eigentlich m=3:= {{30f,2f} {20f, 4f},{10f, 6f}}
        batchNormalizationLayer.forward(EPSILON, miniBatches);
        assertArrayEquals(batchNormalizationLayer.getAverage(miniBatches), new float[]{20f,4f}, DELTA);
        assertArrayEquals(
                batchNormalizationLayer.getVariance(new float[][]{miniBatches[0]},
                        batchNormalizationLayer.getAverage(miniBatches[0])
                ),
                new float[]{66.667f},
                DELTA);
        assertArrayEquals(
                batchNormalizationLayer.getVariance(new float[][]{miniBatches[1]},
                        batchNormalizationLayer.getAverage(miniBatches[1])
                ),
                new float[]{2.667f},
                DELTA);
        float[][] normed= batchNormalizationLayer.getNormalization( miniBatches,
                batchNormalizationLayer.getAverage(miniBatches),
                batchNormalizationLayer.getVariance(miniBatches,
                        batchNormalizationLayer.getAverage(miniBatches)),
                EPSILON);
        assertArrayEquals(normed[0], new float[]{1.225f, 0f, -1.225f}, DELTA);                                          //Die normalisierten Werte des 1. Features
        assertArrayEquals(normed[1], new float[]{-1.225f, 0f, 1.225f}, DELTA);                                          //Die normalisierten Werte des 2. Features

        float[][] dy= new float[][]{{-1f, 2f, -2f}, {0.42f, 0.2f, 0.42f}};                                                //Eingehender Gradient vom Loss (dy)

        assertArrayEquals(
                batchNormalizationLayer.backpropagate(
                        miniBatches[0].length, miniBatches[0], BETA, GAMMA, EPSILON, dy[0]
                )[0],
                new float[]{-0.142f,0.286f,-0.142f},
                DELTA
        );
        assertArrayEquals(
                batchNormalizationLayer.backpropagate(
                        miniBatches[1].length, miniBatches[1], BETA, GAMMA, EPSILON, dy[1]
                )[0],
                new float[]{0.045f,-0.089f,0.045f},
                DELTA
        );
    }

    @Test
    void testBatchNormalizationForwardingBackpropagationMitZweiFeatures() {
        float[][] miniBatches= new float[][]{{2f, 4f, 6f},{10f, 20f, 30f}};
        batchNormalizationLayer.forward(EPSILON, miniBatches);
        assertArrayEquals(batchNormalizationLayer.getAverage(miniBatches), new float[]{4f,20f}, DELTA);
        assertArrayEquals(
                batchNormalizationLayer.getVariance(new float[][]{miniBatches[0]},
                        batchNormalizationLayer.getAverage(miniBatches[0])
                ),
                new float[]{2.667f},
                DELTA);
        assertArrayEquals(
                batchNormalizationLayer.getVariance(new float[][]{miniBatches[1]},
                        batchNormalizationLayer.getAverage(miniBatches[1])
                ),
                new float[]{66.667f},
                DELTA);
        float[][] normed= batchNormalizationLayer.getNormalization( miniBatches,
                batchNormalizationLayer.getAverage(miniBatches),
                batchNormalizationLayer.getVariance(miniBatches,
                        batchNormalizationLayer.getAverage(miniBatches)),
                EPSILON);
        assertArrayEquals(normed[0], new float[]{-1.225f, 0f, 1.225f}, DELTA);                                          //Die normalisierten Werte des 1. Features
        assertArrayEquals(normed[1], new float[]{-1.225f, 0f, 1.225f}, DELTA);                                          //Die normalisierten Werte des 2. Features

        float[][] dy= new float[][]{{1f, -1f, 2f}, {0.1f, -0.2f, 0.3f}};                                                //Eingehender Gradient vom Loss (dy)
        assertArrayEquals(
                batchNormalizationLayer.backpropagate(
                        miniBatches[0].length, miniBatches[0], BETA, GAMMA, EPSILON, dy[0]
                )[0],
                new float[]{0.51f,-1.021f,0.51f},
                DELTA
        );
        assertArrayEquals(
                batchNormalizationLayer.backpropagate(
                        miniBatches[1].length, miniBatches[1], BETA, GAMMA, EPSILON, dy[1]
                )[0],
                new float[]{0.016f,-0.033f,0.016f},
                DELTA
        );
    }

    @Test
    void testBatchNormalizationForwardingBackpropagationB() {                                                          //https://www.google.com/search?q=lstm+batch+normalization+backpropagation+layer+numerisches+rechenbeispiel+mit+mini-batch+m%3D3&client=firefox-b-d&hs=6PvU&sca_esv=3eb9bc99a534cb7d&fbs=ADc_l-YGrpJMQtvjQ6h14rj-dfIrbPkd_Upq68wJVnEIgo2Pwxu679PACFfTKv4n_1_FsyWRtH6VXdFUsrvx5e4Eax_Xgq10hasdmEAWZSPWsvmYCxkgxhgAAEJUbue-qZnt86OK4vP6UejxniOK-dq6UFTtv5LiMcQFF9beBU00mOoHg8GHTk3CzOojQQPhHrd-1Qm9JKHi&aep=1&ntc=1&sa=X&ved=2ahUKEwjn8qjLv72TAxUKgP0HHTK-HsgQ2J8OegQIDxAE&biw=1486&bih=731&dpr=1.25&mstk=AUtExfDggd6fQRXQAqz_Q7wPe-RZkBXEo6aimVzpEQZeKYTYLw7-GxFV8fZ9AZIsE4TJ0RoiKKp3Xdo8ultV6aHGHLupDgrrHWf9PF4ZxAUgHOJ6EeNlPjKqBVZr9dPCm51I7SWPLAPpef2DPW9ct-4G6_UYWoQQugYpZPU&csuir=1&sei=0WrKaZr2N8eE9u8PmJrJuQc&udm=50
        float[] miniBatch= new float[]{2f, 4f, 10f};
        batchNormalizationLayer.forward(EPSILON, miniBatch);
        assertArrayEquals(batchNormalizationLayer.getAverage(miniBatch), new float[]{5.33333f}, DELTA);
        assertArrayEquals(batchNormalizationLayer.getVariance(  new float[][]{miniBatch},
                batchNormalizationLayer.getAverage(miniBatch)),
            new float[]{11.555556f},
            DELTA);
        float[][] normed= batchNormalizationLayer.getNormalization( new float[][]{miniBatch},
            batchNormalizationLayer.getAverage(miniBatch),
            batchNormalizationLayer.getVariance(new float[][]{miniBatch},
                batchNormalizationLayer.getAverage(miniBatch)),
            EPSILON);
        assertArrayEquals(normed[0], new float[]{-0.98058068f, -0.39223227f, 1.37281295f}, DELTA);

        //Angenommen, die zurückgeführten Gradienten vom nächsten Layer sind
        float[][] dLdy= new float[][]{{1.0f, 5.0f, 2.0f}};
        assertArrayEquals(
            batchNormalizationLayer.backpropagate(
                miniBatch.length, miniBatch, BETA, GAMMA, EPSILON, dLdy[0]
            )[0],
            new float[]{-0.50914766f,0.67886354f,-0.16971589f},
            DELTA
        );
    }

    @Test
    void testBatchNormalizationForwardingBackpropagationA() {                                                          //https://www.google.com/search?q=lstm+batch+normalization+backpropagation+layer+numerisches+rechenbeispiel+mit+mini-batch+m%3D3&client=firefox-b-d&hs=6PvU&sca_esv=3eb9bc99a534cb7d&fbs=ADc_l-YGrpJMQtvjQ6h14rj-dfIrbPkd_Upq68wJVnEIgo2Pwxu679PACFfTKv4n_1_FsyWRtH6VXdFUsrvx5e4Eax_Xgq10hasdmEAWZSPWsvmYCxkgxhgAAEJUbue-qZnt86OK4vP6UejxniOK-dq6UFTtv5LiMcQFF9beBU00mOoHg8GHTk3CzOojQQPhHrd-1Qm9JKHi&aep=1&ntc=1&sa=X&ved=2ahUKEwjn8qjLv72TAxUKgP0HHTK-HsgQ2J8OegQIDxAE&biw=1486&bih=731&dpr=1.25&mstk=AUtExfDs-YEqGes-gA8WHq9B7xX71gEWrUPXbBqXhgNsZDei9EemXXSFVshKqVLxLd1Ttf_b1MNUgbChWr0u-caydNHmvD056CH_-RoP0tX7fSb8nm0-Y_31t3ngusUZaKv6bsTQ_7nB8quRLN5nAy-kefKYlTvxEk-PcpM&csuir=1&sei=qz3GafuqCK6K9u8P_vrO0Aw&udm=50
        float[] miniBatch= new float[]{1f, 3f, 5f};
        batchNormalizationLayer.forward(EPSILON, miniBatch);
        assertArrayEquals(batchNormalizationLayer.getAverage(miniBatch), new float[]{3}, DELTA);
        assertArrayEquals(batchNormalizationLayer.getVariance(  new float[][]{miniBatch},
            batchNormalizationLayer.getAverage(miniBatch)),
            new float[]{2.667f},
            DELTA);
        float[][] normed= batchNormalizationLayer.getNormalization( new float[][]{miniBatch},
            batchNormalizationLayer.getAverage(miniBatch),
            batchNormalizationLayer.getVariance(new float[][]{miniBatch},
                        batchNormalizationLayer.getAverage(miniBatch)),
            EPSILON);
        assertArrayEquals(normed[0], new float[]{-1.225f, 0, 1.225f}, DELTA);

        //Angenommen, die Gradienten vom nächsten Layer sind
        float[][] dLdy= new float[][]{{1.0f, 2.0f, 3.0f}};
        assertArrayEquals(
            batchNormalizationLayer.backpropagate(miniBatch.length, miniBatch, BETA, GAMMA, EPSILON, dLdy[0])[0],
            new float[]{0.0f,0.0f,0.0f},
            DELTA
        );
    }

    @Test
    void testBatchNormalizationBackpropagation() {
        float[] miniBatch= new float[]{-0.96f, 1.32f, 0.09f};
        float[][] dL= new float[][]{{-0.14f, -0.92f, 0.6f}};
        batchNormalizationLayer.backpropagate(miniBatch.length, miniBatch, BETA, GAMMA, EPSILON, dL[0]);
        assertArrayEquals(
            batchNormalizationLayer.backpropagate(miniBatch.length, miniBatch, BETA, GAMMA, EPSILON, dL[0])[0],
            new float[]{-0.4234f,-0.361f,0.785f},
            DELTA
        );
    }

    @Test
    void testBatchNormalizationTransformation() {
        float[] miniBatch= new float[]{2f,4f,6f};
        float[][] dL= new float[][]{{0.5f, 0.2f, -0.1f}};
        batchNormalizationLayer.backpropagate(miniBatch.length, miniBatch, BETA, GAMMA, EPSILON, dL[0]);
        float[][]normalizations=
            batchNormalizationLayer.getNormalization(    //Normalisierung
                new float[][]{miniBatch},
                batchNormalizationLayer.getAverage(   miniBatch),
                batchNormalizationLayer.getVariance(
                    new float[][]{ miniBatch},
                    batchNormalizationLayer.getAverage(   miniBatch)
                ),
                EPSILON
        );
        assertArrayEquals(
            batchNormalizationLayer.getTransformations(normalizations, BETA, GAMMA)[0],
            new float[]{-1.224f,0,1.224f},
            DELTA
        );
    }

    @Test
    void testBatchNormalizationLayerF() {
        float[] miniBatch= new float[]{2f, 4f, 6f};
        //Forwardpass
        assertArrayEquals(
            batchNormalizationLayer.getAverage(miniBatch),                  //Mittelwert
            new float[]{4f} , DELTA
        );
        assertArrayEquals(
            batchNormalizationLayer.getVariance(                            //Varianz
                new float[][]{miniBatch},
                batchNormalizationLayer.getAverage(miniBatch)
            ),
            new float[]{2.666f} , DELTA
        );
        float[][]normalizations=
            batchNormalizationLayer.getNormalization(    //Normalisierung
                new float[][]{miniBatch},
                batchNormalizationLayer.getAverage(miniBatch),
                batchNormalizationLayer.getVariance(
                    new float[][]{miniBatch},
                    batchNormalizationLayer.getAverage(miniBatch)
                ),
                EPSILON
            );
        assertArrayEquals(new float[]{-1.224f,0f,1.224f} , normalizations[0] , DELTA);
        float GAMMA=0.5f, BETA=0.3f;
        assertArrayEquals(
            batchNormalizationLayer.scaleAndShift(normalizations, GAMMA, BETA)[0],
            new float[]{-0.3125f, 0.3f, 0.9125f} ,
            DELTA
        );
    }

    @Test
    void testBatchNormalizationLayerD() {
        //Form der Eingabedaten: (2, 4, 6) - (Samples, Timesteps, Features)
        assertArrayEquals(
            batchNormalizationLayer.getAverage(new float[]{2.0f, 4.0f, 6.0f,}),  //Zeitpunkt1
            new float[]{4f},
            DELTA
        );
        assertArrayEquals(
            batchNormalizationLayer.getVariance(
                new float[][]{ new float[]{2.0f, 4.0f, 6.0f,} },
                batchNormalizationLayer.getAverage(new float[]{2.0f, 4.0f, 6.0f,})
            ),
            new float[]{2.666f},
            DELTA
        );
        float[][]normalized=
            batchNormalizationLayer.getNormalization(
                new float[][]{new float[]{2.0f, 4.0f, 6.0f,}},
                batchNormalizationLayer.getAverage(new float[]{2.0f, 4.0f, 6.0f,}),
                batchNormalizationLayer.getVariance(
                    new float[][]{ new float[]{2.0f, 4.0f, 6.0f,} },
                    batchNormalizationLayer.getAverage(new float[]{2.0f, 4.0f, 6.0f,})
                ),
                EPSILON
            );
        assertArrayEquals(new float[]{-1.224f,0f,1.224f} , normalized[0] , DELTA);
    }
    @Test
    void testBatchNormalizationLayerC() {
        //Form der Eingabedaten: (3, 3, 1) - (Samples, Timesteps, Features)
        assertArrayEquals(
            batchNormalizationLayer.getAverage(
                new float[]{1.0f, 2.0f, 3.0f},  //Zeitpunkt1
                new float[]{2.0f, 3.0f, 4.0f},  //Zeitpunkt2
                new float[]{3.0f, 4.0f, 5.0f}   //Zeitpunkt3
            ),
            new float[]{2f,3f,4f} ,
            DELTA
        );
        assertArrayEquals(
            batchNormalizationLayer.getVariance(
                new float[][]{
                    new float[]{1.0f, 2.0f, 3.0f},  //Zeitpunkt1
                    new float[]{2.0f, 3.0f, 4.0f},  //Zeitpunkt2
                    new float[]{3.0f, 4.0f, 5.0f},  //Zeitpunkt3
                },
                batchNormalizationLayer.getAverage(new float[]{1.0f, 2.0f, 3.0f},  //Zeitpunkt1
                    new float[]{2.0f, 3.0f, 4.0f},  //Zeitpunkt2
                    new float[]{3.0f, 4.0f, 5.0f}   //Zeitpunkt3
                )
            ),
            new float[]{0.666f,0.666f,0.666f} , DELTA
        );
    }
    @Test
    void testBatchNormalizationLayerB() {
        assertArrayEquals(
            batchNormalizationLayer.getAverage(new float[]{2.0f, 4.0f, 6.0f,} ),  //Zeitpunkt1
            new float[]{4f},
            DELTA
        );
        assertArrayEquals(
            batchNormalizationLayer.getVariance(
                new float[][]{
                    new float[]{2.0f, 4.0f, 6.0f,}
                },
                batchNormalizationLayer.getAverage(new float[]{2.0f, 4.0f, 6.0f,})
            ),
            new float[]{2.666f},
            DELTA
        );
        float[][]zValues=
            batchNormalizationLayer.getNormalization(
                new float[][]{new float[]{2.0f, 4.0f, 6.0f,}},
                batchNormalizationLayer.getAverage(new float[]{2.0f, 4.0f, 6.0f,}),
                batchNormalizationLayer.getVariance(
                    new float[][]{ new float[]{2.0f, 4.0f, 6.0f,} },
                    batchNormalizationLayer.getAverage(new float[]{2.0f, 4.0f, 6.0f,})
                ),
                EPSILON
            );
        assertArrayEquals(new float[]{-1.224f,0f,1.224f} , zValues[0] , DELTA);
    }

    @Test
    void testBatchNormalizationLayerA() {
        assertArrayEquals(
            batchNormalizationLayer.getAverage(
                new float[]{1.0f,    3f},   //Zeitpunkt1
                new float[]{2.0f,    4f},   //Zeitpunkt2
                new float[]{0.5f,   1.5f}   //Zeitpunkt3
            ),
            new float[]{2f,3f,1f},
            DELTA
        );

        assertArrayEquals(
            batchNormalizationLayer.getVariance(
                new float[][]{
                    new float[]{1.0f,    3f},   //Zeitpunkt1
                    new float[]{2.0f,    4f},   //Zeitpunkt2
                    new float[]{0.5f,   1.5f}   //Zeitpunkt3
                },
                batchNormalizationLayer.getAverage(
                    new float[]{1.0f,    3f},   //Zeitpunkt1
                    new float[]{2.0f,    4f},   //Zeitpunkt2
                    new float[]{0.5f,   1.5f}   //Zeitpunkt3
                )
            ),
        new float[]{1.0f,1f,0.25f} , DELTA);

        float[][]zValues=
            batchNormalizationLayer.getNormalization(
                new float[][]{
                    new float[]{1.0f,    3f},
                    new float[]{2.0f,    4f},
                    new float[]{0.5f,   1.5f}
                },
                batchNormalizationLayer.getAverage(
                    new float[]{1.0f,    3f},
                    new float[]{2.0f,    4f},
                    new float[]{0.5f,   1.5f}
                ),
                batchNormalizationLayer.getVariance(
                    new float[][]{
                        new float[]{1.0f,    3f},
                        new float[]{2.0f,    4f},
                        new float[]{0.5f,   1.5f}
                    },
                    batchNormalizationLayer.getAverage( new float[]{1.0f,    3f},
                        new float[]{2.0f,    4f},
                        new float[]{0.5f,   1.5f}
                    )
                ),
                EPSILON
            );
        assertArrayEquals( zValues[0] , new float[]{-1.0f,1.0f} , DELTA);
        assertArrayEquals( zValues[1] , new float[]{-1.0f,1.0f} , DELTA);
        assertArrayEquals( zValues[2] , new float[]{-1.0f,1.0f} , DELTA);
    }
}
