package lstm2603200945;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Test_DenseLayer {
    @Test
    void testDenseLayer() {
        float DELTA= 1e-3f;
        float[][]weights= new float[][]{{0.2f, 0.8f},
                                        {0.5f, -0.1f},
                                        {0.1f, 0.3f}};
        DenseLayer denseLayer= new DenseLayer(new float[]{1.0f, 0.5f, -0.2f},
                                              weights, new float[]{0.1f, 0.2f, 0.3f});
        assertArrayEquals( denseLayer.getOutput() ,
                new float[]{0.53f,0.89f} , DELTA);                    //Ohne Aktivierungsfunktion[sigmoid]
//                new float[]{0.6294831f,0.7088902f} , DELTA);            //Mit Aktivierungsfunktion[sigmoid]
    }
    @Test
    void testDenseLayerOverloadConstructor() {
        DenseLayer denseLayer= new DenseLayer(new float[]{1.0f, 0.5f, -0.2f}, 3);
        assertTrue(true);
    }

    @Test
    void anotherDenseLayerTest() {
        float[]input= new float[]{1.0f, 0.5f, -0.2f};
        float[][]weightMatrix= new float[][]   {{0.2f, 0.8f},
                                                {0.5f, -0.1f},
                                                {0.1f, 0.3f}};
        float[]bias= new float[]{0.1f, 0.2f};
        DenseLayer denseLayer= new DenseLayer(input, weightMatrix, bias);

        assertTrue(denseLayer.getOutput()[0]==0.53f);
        assertTrue(denseLayer.getOutput()[1]==0.89f);
    }
}
