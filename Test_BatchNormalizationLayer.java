package lstm2603200945;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class Test_BatchNormalizationLayer {
    final BatchNormalizationLayer batchNormalizationLayer=
            new BatchNormalizationLayer(new float[]{1.0f,    3f},  //Zeitpunkt1
                                        new float[]{2.0f,    4f},  //Zeitpunkt2
                                        new float[]{0.5f,   1.5f}); //Zeitpunkt3
    @Test
    void testBatchNormalizationLayer() {
        float DELTA= 1e-3f;
        assertArrayEquals(batchNormalizationLayer.getAverage(   new float[]{1.0f,    3f},  //Zeitpunkt1
                                                                new float[]{2.0f,    4f},  //Zeitpunkt2
                                                                new float[]{0.5f,   1.5f}),//Zeitpunkt3
                                                                new float[]{2f,3f,1f} , DELTA);

        assertArrayEquals(  batchNormalizationLayer.getVariance(
                                new float[][]{
                                        new float[]{1.0f,    3f},
                                        new float[]{2.0f,    4f},
                                        new float[]{0.5f,   1.5f}},
                                batchNormalizationLayer.getAverage( new float[]{1.0f,    3f},
                                                                    new float[]{2.0f,    4f},
                                                                    new float[]{0.5f,   1.5f})
                            ),
                            new float[]{1.0f,1f,0.25f} , DELTA);

        float[][]zValues=   batchNormalizationLayer.getZValue(
                                new float[][]{
                                        new float[]{1.0f,    3f},
                                        new float[]{2.0f,    4f},
                                        new float[]{0.5f,   1.5f}},
                                batchNormalizationLayer.getAverage( new float[]{1.0f,    3f},
                                        new float[]{2.0f,    4f},
                                        new float[]{0.5f,   1.5f}),
                                batchNormalizationLayer.getVariance(
                                        new float[][]{
                                                new float[]{1.0f,    3f},
                                                new float[]{2.0f,    4f},
                                                new float[]{0.5f,   1.5f}},
                                        batchNormalizationLayer.getAverage( new float[]{1.0f,    3f},
                                                new float[]{2.0f,    4f},
                                                new float[]{0.5f,   1.5f})
                                )
                            );
        assertArrayEquals( zValues[0] ,
                new float[]{-1.0f,1.0f} , DELTA);
        assertArrayEquals( zValues[1] ,
                new float[]{-1.0f,1.0f} , DELTA);
        assertArrayEquals( zValues[2] ,
                new float[]{-1.0f,1.0f} , DELTA);
    }
}
