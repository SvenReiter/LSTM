package lstm2604131306;

import java.util.ArrayList;
import java.util.List;

public class Start {
    public static void main(String[] args) {
        //LSTM-Zellen einer Schicht nutzen selbe Gewichte/Biases über alle Zeitschritte einer ZellenReihe/Sequenz hinweg.
        /*Shared Weights bei LSTM-Netzwerken bedeuten, dass dieselben Gewichtungsmatrizen für jeden Zeitschritt
         *in einer Eingabesequenz verwendet werden. */
        //Angenommen zu bearbeiten sei ZellenReihe(Sequenz) 1 mit diesen Gewichtsmatrizen
        float[][] wF= new float[][]{{0.813f,-0.487f,0.02f,-0.778f,0.418f},{-0.708f, 0.006f, 0.856f, -0.106f, -0.872f}};
        float[][] wI= new float[][]{{-0.209f,-0.14f,0.031f,0.226f,0.696f},{0.101f, -0.435f, -0.406f, -0.796f, 0.324f}};
        float[][] wC= new float[][]{{-0.901f,-0.877f,-0.413f,0.16f,-0.775f},{-0.196f,0.077f, 0.769f, -0.567f, -0.905f}};
        float[][] wO= new float[][]{{0.668f,-0.605f,-0.402f,-0.691f,-0.486f},{0.613f,0.875f, 0.549f, -0.623f, 0.262f}};
        float[][] wY= new float[][]{{0.32f,-0.172f},{0.449f,0.349f},{0.914f,0.371f}};
        float[][] wF_sec= new float[][]{{0.813f, -0.487f, 0.02f, -0.778f},{-0.708f, 0.006f, 0.856f, -0.106f}};
        float[][] wI_sec= new float[][]{{-0.209f, -0.14f, 0.031f, 0.226f},{0.101f, -0.435f, -0.406f, -0.796f}};
        float[][] wC_sec= new float[][]{{-0.901f, -0.877f, -0.413f, 0.16f},{-0.196f, 0.077f, 0.769f, -0.567f}};
        float[][] wO_sec= new float[][]{{0.668f, -0.605f, -0.402f, -0.691f},{0.613f, 0.875f, 0.549f, -0.623f}};
        float[] bF= new float[]{1.f, 1.f};      //Die Biases des Forget Gates sollten auf einen positiven Wert werden.
        float[] bI= new float[]{0.f, 0.f};      //Die Biases für die anderen Gates
        float[] bC= new float[]{0.f, 0.f};      //  (Input, Output) und den Zellzustand werden
        float[] bO= new float[]{0.f, 0.f};      //  typischerweise auf 0 gesetzt.

        /* Architektur wie auf DenseLstmArchitectureConstistsAlsoBatchNormailzationLayer.png gezeigt realisieren
           Es ist so, dass der Input von der ersten cellRow [Xt] entspr.
           Aber der Input von jeder weiteren cellRow ist das Resultat des untergeordneten <BatchNormalizationLayer>
           Die Konsistenz der <LSTM_cell>s in LSTM_cells[] muss verändert werden!
           Nur die <LSTM_cell>s der ersten CellRow/CellLayer erhalten einen konkreten
           Input Xt zum Zeitpunkt ihrer Definition.
           Die Inputs der <LSTM_cell>s aus den übrigen CellRows/CellLayers erfolgen
           über das Resultat der jeweiligen <BatchNormalizationLayer>.
           Genauso wie in DenseLstmArchitectureConstistsAlsoBatchNormailzationLayer.png abgebildet.
         */
        final int NUMBER_CELLS_PER_CHAIN= 5, NUMBER_CELL_CHAIN_LAYERS= 3, NUMBER_FEATURES= 2;
        List<List<LSTM_cell>> LSTM_cells_layers= new ArrayList<>();
        List<LSTM_cell> LSTM_cells_firstLayer = new ArrayList<>();
        BatchNormalizationLayer batchNormalizationLayer_first=
                new BatchNormalizationLayer(NUMBER_CELLS_PER_CHAIN,NUMBER_FEATURES);
        BatchNormalizationLayer batchNormalizationLayer_second=
                new BatchNormalizationLayer(NUMBER_CELLS_PER_CHAIN,NUMBER_FEATURES);
        for (int i = 0; i < NUMBER_CELLS_PER_CHAIN; i++) {
            LSTM_cells_firstLayer.add(
                    new LSTM_cell(
                            new float[]{1f,0f},                                                                         //Input CS(t-1)
                            new float[]{0f,0f},                                                                         //Input HS(t-1)
                            new float[]{1f,0f,0f},                                                                      //Input x(t)
                            new float[]{0,1,0},
                            wF, wI, wC, wO,
                            bF, bI, bC, bO
                    )
            );
        }
        for (int i = 0; i < NUMBER_CELLS_PER_CHAIN; i++) {
            if(i==0 && LSTM_cells_firstLayer.size()>0 && (i+1)<=(LSTM_cells_firstLayer.size()-1)) {                     //Wenn erste LSTM-cell in Sequenz=> braucht kein <prevCell>
                LSTM_cells_firstLayer.get(i).prevCell=
                        new LSTM_cell(new float[]{1f, 0f}, new float[]{0f, 0f}, new float[]{1f,0f,0f},
                                      new float[]{0,1,0}, wF, wI, wC, wO, bF, bI, bC, bO);
                LSTM_cells_firstLayer.get(i).nextCell= LSTM_cells_firstLayer.get(i+1);
            }else if(i>0 && i==(NUMBER_CELLS_PER_CHAIN-1)) {                                                            //Wenn letzte LSTM-cell in Sequenz=> braucht kein <nextcell>
                LSTM_cells_firstLayer.get(i).prevCell= LSTM_cells_firstLayer.get(i-1);
            }else {                                                                                                     //Ansonsten=> braucht <prevCell> und <nextcell>
                LSTM_cells_firstLayer.get(i).nextCell= LSTM_cells_firstLayer.get(i+1);
                LSTM_cells_firstLayer.get(i).prevCell= LSTM_cells_firstLayer.get(i-1);
            }
        }
        LSTM_cells_layers.add(LSTM_cells_firstLayer);
        List<LSTM_cell> LSTM_cells_secondLayer = new ArrayList<>();                                                     //Beschreibt den alle weiteren LSTM-Layer
        for (int i = 0; i < NUMBER_CELLS_PER_CHAIN; i++) {
            LSTM_cells_secondLayer.add(
                    new LSTM_cell(
                            new float[]{1f,0f},                                                                         //Input CS(t-1)
                            new float[]{0f,0f},                                                                         //Input HS(t-1)
                            batchNormalizationLayer_first.getNormalization()[i],                                               //Input x(t)
                            new float[]{0,1,0},
                            wF_sec, wI_sec, wC_sec, wO_sec,
                            bF, bI, bC, bO
                    )
            );
        }
        for (int i = 0; i < NUMBER_CELLS_PER_CHAIN; i++) {
            if(i==0 && LSTM_cells_secondLayer.size()>0 && (i+1)<=(LSTM_cells_secondLayer.size()-1)) {                   //Wenn erste LSTM-cell in Sequenz=> braucht kein <prevCell>
                LSTM_cells_secondLayer.get(i).prevCell= new LSTM_cell(new float[]{1f, 0f}, new float[]{0f, 0f},
                                                        new float[]{1f,0f,0f}, new float[]{0,1,0},
                                                        wF, wI, wC, wO, bF, bI, bC, bO);
                LSTM_cells_secondLayer.get(i).nextCell= LSTM_cells_secondLayer.get(i+1);
            }else if(i>0 && i==(NUMBER_CELLS_PER_CHAIN-1)) {                                                            //Wenn letzte LSTM-cell in Sequenz=> braucht kein <nextcell>
                LSTM_cells_secondLayer.get(i).prevCell= LSTM_cells_secondLayer.get(i-1);
            }else {                                                                                                     //Ansonsten=> braucht <prevCell> und <nextcell>
                LSTM_cells_secondLayer.get(i).nextCell= LSTM_cells_secondLayer.get(i+1);
                LSTM_cells_secondLayer.get(i).prevCell= LSTM_cells_secondLayer.get(i-1);
            }
        }
        LSTM_cells_layers.add(LSTM_cells_secondLayer);

        List<LSTM_cell> LSTM_cells_thirdLayer = new ArrayList<>();                                                      //Beschreibt den alle weiteren LSTM-Layer
        for (int i = 0; i < NUMBER_CELLS_PER_CHAIN; i++) {
            LSTM_cells_thirdLayer.add(
                    new LSTM_cell(
                            new float[]{1f,0f},                                                                         //Input CS(t-1)
                            new float[]{0f,0f},                                                                         //Input HS(t-1)
                            batchNormalizationLayer_second.getNormalization()[i],                                              //Input x(t)
                            new float[]{0,1,0},
                            wF_sec, wI_sec, wC_sec, wO_sec,
                            bF, bI, bC, bO
                    )
            );
        }
        for (int i = 0; i < NUMBER_CELLS_PER_CHAIN; i++) {
            if(i==0 && LSTM_cells_thirdLayer.size()>0 && (i+1)<=(LSTM_cells_thirdLayer.size()-1)) {                     //Wenn erste LSTM-cell in Sequenz=> braucht kein <prevCell>
                LSTM_cells_thirdLayer.get(i).prevCell= new LSTM_cell(new float[]{1f, 0f}, new float[]{0f, 0f},
                                                                     new float[]{1f,0f,0f}, new float[]{0,1,0},
                                                                     wF, wI, wC, wO, bF, bI, bC, bO);
                LSTM_cells_thirdLayer.get(i).nextCell= LSTM_cells_thirdLayer.get(i+1);
            }else if(i>0 && i==(NUMBER_CELLS_PER_CHAIN-1)) {                                                            //Wenn letzte LSTM-cell in Sequenz=> braucht kein <nextcell>
                LSTM_cells_thirdLayer.get(i).prevCell= LSTM_cells_thirdLayer.get(i-1);
            }else {                                                                                                     //Ansonsten=> braucht <prevCell> und <nextcell>
                LSTM_cells_thirdLayer.get(i).nextCell= LSTM_cells_thirdLayer.get(i+1);
                LSTM_cells_thirdLayer.get(i).prevCell= LSTM_cells_thirdLayer.get(i-1);
            }
        }
        LSTM_cells_layers.add(LSTM_cells_thirdLayer);

        float[][]activations_first= new float[LSTM_cells_layers.get(1).size()][NUMBER_FEATURES];
        float[][]activations_second= new float[LSTM_cells_layers.get(0).size()][NUMBER_FEATURES];
        for(List<LSTM_cell> cellLayer: LSTM_cells_layers) {
            for(int i=0; i< cellLayer.size(); i++) {
                cellLayer.get(i).forward(wY);
                if(cellLayer.equals(LSTM_cells_layers.get(0))) {
                    for(int elm=0; elm<cellLayer.get(i).hiddenState.length; elm++) {
                        for(int f=0; f<NUMBER_FEATURES; f++) {
                            activations_first[i][f] = cellLayer.get(i).hiddenState[f];
                        }
                    }
                }else if(cellLayer.equals(LSTM_cells_layers.get(1))) {
                    for(int elm=0; elm<cellLayer.get(i).hiddenState.length; elm++) {
                        for(int f=0; f<NUMBER_FEATURES; f++) {
                            activations_second[i][f] = cellLayer.get(i).hiddenState[f];
                        }
                    }
                }
            }
        }
        float EPSILON= 0f;
        batchNormalizationLayer_first= new BatchNormalizationLayer(EPSILON, activations_first);
        batchNormalizationLayer_second= new BatchNormalizationLayer(EPSILON, activations_second);
        System.out.println();
        /*todo Die Input-Werte bzw. Xt-Werte der LSTM_cells müssen aktualisiert werden nachdem die BatchNormalizationLayer verarbeitet sind
            1.) Forwarding der 1. LSTM_cell-Row
            2.) entspr. BatchNormalizationLayer verarbeitet die Werte zu zValues
            3.) die vorher berechneten zValues bilden den Input für die 2. LSTM_cell-Row
         */
    }
}