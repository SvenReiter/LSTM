package lstm2603200945;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Start {
    public static void main(String[] args) {
        //LSTM-Zellen innerhalb derselben Schicht nutzen dieselben Gewichte/Biases über alle Zeitschritte einer Sequenz hinweg.
        /*Shared Weights bei LSTM-Netzwerken bedeuten, dass dieselben Gewichtungsmatrizen für jeden Zeitschritt
         *in einer Eingabesequenz verwendet werden. */
        //Angenommen zu bearbeiten sei Sequenz 1 mit diesen Gewichtsmatrizen
        float[][] wF= new float[][]{        // hs[].size * x[].size    bzw. hs[].size * (hs[].size+inputX[].size)
                {0.813f, -0.487f, 0.02f, -0.778f, 0.418f},
                {-0.708f, 0.006f, 0.856f, -0.106f, -0.872f}
        };
        float[][] wI= new float[][]{
                {-0.209f, -0.14f, 0.031f, 0.226f, 0.696f},
                {0.101f, -0.435f, -0.406f, -0.796f, 0.324f}
        };
        float[][] wC= new float[][]{
                {-0.901f, -0.877f, -0.413f, 0.16f, -0.775f},
                {-0.196f, 0.077f, 0.769f, -0.567f, -0.905f}
        };
        float[][] wO= new float[][]{
                {0.668f, -0.605f, -0.402f, -0.691f, -0.486f},
                {0.613f, 0.875f, 0.549f, -0.623f, 0.262f}
        };
        float[][] wY= new float[][]{
                {0.32f, -0.172f},
                {0.449f, 0.349f},
                {0.914f, 0.371f}
        };
        float[] bF= new float[]{0.5f, 0.5f};
        float[] bI= new float[]{0.f, 0.f};
        float[] bC= new float[]{0.f, 0.f};
        float[] bO= new float[]{0.f, 0.f};

        /* TODO: 19.03.2026
            Architektur wie auf DenseLstmArchitectureConstistsAlsoBatchNormailzationLayer.png gezeigt realisieren

         */
        final int NUMBER_CELLS_PER_CHAIN= 5, NUMBER_CELL_CHAIN_LAYERS= 3;
        List<List<LSTM_cell>> LSTM_cells_layers= new ArrayList<>();
        for(int j=0; j<NUMBER_CELL_CHAIN_LAYERS; j++) {
            List<LSTM_cell> LSTM_cells = new ArrayList<>();                                                             //Beschreibt ein LSTM-Layer
            for (int i = 0; i < NUMBER_CELLS_PER_CHAIN; i++) {
                LSTM_cells.add(new LSTM_cell(   new float[]{1f,0f},                                                     //Input CS(t-1)
                        new float[]{0f,0f},                                                     //Input HS(t-1)
                        new float[]{1f,0f,0f},                                                  //Input x(t)
                        new float[]{0,1,0},
                        wF, wI, wC, wO,
                        bF, bI, bC, bO));
            }
            for (int i = 0; i < NUMBER_CELLS_PER_CHAIN; i++) {
                if(i==0 && LSTM_cells.size()>0 && (i+1)<=(LSTM_cells.size()-1)) {                       //Wenn erste LSTM-cell in Sequenz=> braucht kein <prevCell>
                    LSTM_cells.get(i).nextCell= LSTM_cells.get(i+1);
                }else if(i>0 && i==(NUMBER_CELLS_PER_CHAIN-1)) {        //Wenn letzte LSTM-cell in Sequenz=> braucht kein <nextcell>
                    LSTM_cells.get(i).prevCell= LSTM_cells.get(i-1);
                }else {                                                 //Ansonsten=> braucht <prevCell> und <nextcell>
                    LSTM_cells.get(i).nextCell= LSTM_cells.get(i+1);
                    LSTM_cells.get(i).prevCell= LSTM_cells.get(i-1);
                }
            }
            LSTM_cells_layers.add(LSTM_cells);
        }




        LSTM_cell cell= new LSTM_cell(  new float[]{1f,0f},                                                             //Input CS(t-1)
                                        new float[]{0f,0f},                                                             //Input HS(t-1)
                                        new float[]{1f,0f,0f},                                                          //Input x(t)
                                        new float[]{0,1,0},
                                        wF, wI, wC, wO,
                                        bF, bI, bC, bO);
        cell.nextCell = new LSTM_cell(  new float[]{0f,0f}, new float[]{0f,0f},
                                        new float[]{1f,0f,0f}, new float[]{0,1,0},
                                        wF, wI, wC, wO,
                                        bF, bI, bC, bO);
        cell.prevCell = new LSTM_cell(  new float[]{0f,0f}, new float[]{1f,0f},
                                        new float[]{1f,0f,0f}, new float[]{0,1,0},
                                        wF, wI, wC, wO,
                                        bF, bI, bC, bO);
        float resTotalLoss= 0, totalLossBefore= 0;
        cell.prevCell.cellState= new float[]{1f,0f};
        cell.nextCell.hiddenState= new float[]{0f,1f};
        cell.forward(wY);
        /*Here: <node.CellState>    := CS(t)
                <node.HiddenState>  := HS(t)
                <node.z>            := Zt
                <node.yHat>         := Softmax(Yt_hat)
           https://www.youtube.com/watch?v=fWXIkBHWPHY&lc=Ugz2MoWWc8N3gRCSflZ4AaABAg(10:11)
         */

        cell.calcLoss(cell.yHat, cell.y);                                                                               //The loss becomes delLdelZ[t]

        System.out.println( "Walkthrough0 Loss: "+ Arrays.toString(cell.delLdelZ)+
                "\n\tCellState: "+ Arrays.toString(cell.cellState) +
                "\tHiddenState: "+ Arrays.toString(cell.hiddenState));
        System.out.print("\tZ(t):= "+Arrays.toString(cell.z));
        System.out.println("\tyHat:= "+Arrays.toString(cell.yHat));
        for(float f:cell.delLdelZ) resTotalLoss+=Math.abs(f);
        System.out.println("\t  Resultierender Gesamtfehler:"+ resTotalLoss);
        final int LOOPS= 5;
        for(int i=1; i<LOOPS; i++) {
            cell.backpropagate(cell.xt, cell.Xt, wY);
            cell.updatingWeights(wF, wI, wC, wO, wY);                                                                       //updating weights by delLdelW*-partialDerivatives
            cell.forward(wY);
        /*Here: <node.CellState>    := CS(t)
                <node.HiddenState>  := HS(t)
                <node.z>            := Zt
                <node.yHat>         := Softmax(Yt_hat)
           https://www.youtube.com/watch?v=fWXIkBHWPHY&lc=Ugz2MoWWc8N3gRCSflZ4AaABAg(10:11)
         */

            cell.calcLoss(cell.yHat, cell.y);                                                                               //The loss becomes delLdelZ[t]

            System.out.println("\nWalkthrough" + i + " Loss: " + Arrays.toString(cell.delLdelZ) +
                    "\n\tCellState: " + Arrays.toString(cell.cellState) +
                    "\tHiddenState: " + Arrays.toString(cell.hiddenState));
            System.out.print("\tZ(t):= " + Arrays.toString(cell.z));
            System.out.println("\tyHat:= " + Arrays.toString(cell.yHat));
            totalLossBefore= resTotalLoss;
            resTotalLoss= 0;
            for (float f : cell.delLdelZ) resTotalLoss += Math.abs(f);
            System.out.println("\t  Resultierender Gesamtfehler:" + resTotalLoss);
            if(resTotalLoss>= totalLossBefore) {
                System.err.println("Das hat nicht geklappt");
                return;
            }
        }
        System.out.println("Der Fehler hat jede Iteration weiter abgenommen");
    }
}