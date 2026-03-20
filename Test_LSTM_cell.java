package lstm2603200945;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class Test_LSTM_cell {
    final float[][] wF= new float[][]{                                                                                  // hs[].size * x[].size    bzw. hs[].size * (hs[].size+inputX[].size)
            {0.813f, -0.487f, 0.02f, -0.778f, 0.418f},
            {-0.708f, 0.006f, 0.856f, -0.106f, -0.872f}
    };
    final float[][] wI= new float[][]{
            {-0.209f, -0.14f, 0.031f, 0.226f, 0.696f},
            {0.101f, -0.435f, -0.406f, -0.796f, 0.324f}
    };
    final float[][] wC= new float[][]{
            {-0.901f, -0.877f, -0.413f, 0.16f, -0.775f},
            {-0.196f, 0.077f, 0.769f, -0.567f, -0.905f}
    };
    final float[][] wO= new float[][]{
            {0.668f, -0.605f, -0.402f, -0.691f, -0.486f},
            {0.613f, 0.875f, 0.549f, -0.623f, 0.262f}
    };
    final float[][] wY= new float[][]{
            {0.32f, -0.172f},
            {0.449f, 0.349f},
            {0.914f, 0.371f}
    };

    final LSTM_cell node= new LSTM_cell( new float[]{1f,0f}, new float[]{0f,0f},
                                         new float[]{1f,0f,0f}, new float[]{0,1,0}, wF, wI, wC, wO,
                                         new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                                         new float[]{0.f, 0.f}, new float[]{0.f, 0.f});

    final LSTM_cell nextNode= new LSTM_cell(new float[]{1f,0f}, new float[]{0f,0f},
                                            new float[]{1f,0f,0f}, new float[]{0,1,0}, wF, wI, wC, wO,
                                            new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                                            new float[]{0.f, 0.f}, new float[]{0.f, 0.f});

    final LSTM_cell prevNode= new LSTM_cell(new float[]{1f,0f}, new float[]{0f,0f},
                                            new float[]{1f,0f,0f}, new float[]{0,1,0}, wF, wI, wC, wO,
                                            new float[]{0.f, 0.f}, new float[]{0.f, 0.f},
                                            new float[]{0.f, 0.f}, new float[]{0.f, 0.f});

    @Test
    void testResultingWeightAdjustmentsAfterForwarding() {
        float DELTA= 1e-3f;
        node.forward(wY);
        assertArrayEquals( new float[]{0.011f, 0.109f, 0.168f}, node.z, DELTA);
        assertArrayEquals( new float[]{0.305f, 0.337f, 0.358f}, node.yHat, DELTA);
        node.calcLoss(node.yHat, node.y);
        assertArrayEquals( new float[]{0.305f, -0.663f, 0.358f}, node.delLdelZ, DELTA);
    }

    @Test
    void testResultingWeightAdjustmentsAfterBackpropagation() {
        node.nextCell = nextNode;
        node.prevCell = prevNode;
        float DELTA= 1e-3f;
        node.forward(wY);
        node.calcLoss(node.yHat, node.y);
        node.backpropagate(node.xt, node.Xt, wY);
        assertEquals(node.getDelLdelW('o')[0][2], 0.009, DELTA);
        assertEquals(node.getDelLdelW('o')[1][2], -0.0087, DELTA);
//        assertEquals(lstm.getDelLdelW('f')[0][2], 0.011, DELTA);                                                      //Evt ergab sich Änderung weil ich bei Biases einbeziehe
        assertEquals(node.getDelLdelW('f')[0][2], 0.245, DELTA);
        assertEquals(node.getDelLdelW('f')[1][2], 0.0, DELTA);
        assertEquals(node.getDelLdelW('i')[0][2], -0.004, DELTA);
        assertEquals(node.getDelLdelW('i')[1][2], -0.014, DELTA);
        assertEquals(node.getDelLdelW('c')[0][2], 0.02, DELTA);
        assertEquals(node.getDelLdelW('c')[1][2], -0.024, DELTA);

        assertEquals(node.getDelLdelW('y')[0][0], 0.036, DELTA);
        assertEquals(node.getDelLdelW('y')[0][1], 0.049, DELTA);
        assertEquals(node.getDelLdelW('y')[1][0], -0.079, DELTA);
        assertEquals(node.getDelLdelW('y')[1][1], -0.106, DELTA);
        assertEquals(node.getDelLdelW('y')[2][0], 0.043, DELTA);
        assertEquals(node.getDelLdelW('y')[2][1], 0.057, DELTA);
    }

    @Test
    public void checksDecreasingLossesOverTime() {
        node.nextCell = nextNode;
        node.prevCell = prevNode;
        float resTotalLossBefore,resTotalLoss= 0;
        node.forward(wY);
        node.calcLoss(node.yHat, node.y);                                                                               //The loss becomes delLdelZ[t]
        for(float f:node.delLdelZ) resTotalLoss+=Math.abs(f);
        resTotalLossBefore= resTotalLoss;
        resTotalLoss=0;
        node.forward(wY);
        node.calcLoss(node.yHat, node.y);                                                                               //The loss becomes delLdelZ[t]
        for(float f:node.delLdelZ) resTotalLoss+=Math.abs(f);
        assertTrue(resTotalLossBefore> resTotalLoss);
        resTotalLossBefore= resTotalLoss;
        resTotalLoss=0;
        node.forward(wY);
        node.calcLoss(node.yHat, node.y);                                                                               //The loss becomes delLdelZ[t]
        for(float f:node.delLdelZ) resTotalLoss+=Math.abs(f);
        assertTrue(resTotalLossBefore> resTotalLoss);
        resTotalLossBefore= resTotalLoss;
        resTotalLoss=0;
        node.forward(wY);
        node.calcLoss(node.yHat, node.y);                                                                               //The loss becomes delLdelZ[t]
        for(float f:node.delLdelZ) resTotalLoss+=Math.abs(f);
        assertTrue(resTotalLossBefore> resTotalLoss);
        resTotalLossBefore= resTotalLoss;
        resTotalLoss=0;
        node.forward(wY);
        node.calcLoss(node.yHat, node.y);                                                                               //The loss becomes delLdelZ[t]
        for(float f:node.delLdelZ) resTotalLoss+=Math.abs(f);
        assertTrue(resTotalLossBefore> resTotalLoss);
        resTotalLossBefore= resTotalLoss;
        resTotalLoss=0;
        node.forward(wY);
        node.calcLoss(node.yHat, node.y);                                                                               //The loss becomes delLdelZ[t]
        for(float f:node.delLdelZ) resTotalLoss+=Math.abs(f);
        assertTrue(resTotalLossBefore> resTotalLoss);
    }
}
