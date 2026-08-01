package lstm2608010711_BerechneteResultateVonJedemTimestepSindWeggespeichert;

public class Start {
    public static void main(String[] args) {
        float[]bf=new float[]{0.1f,0.1f},bi=new float[]{0.1f,0.1f},bc=new float[]{0.1f,0.1f},bo=new float[]{0.1f,0.1f};
        float[][]wf= new float[][]{{0.5f, 0f}, {0f, 0.5f}}; float[][]wi= new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][]wc= new float[][]{{0.5f, 0f}, {0f, 0.5f}}; float[][]wo= new float[][]{{0.5f, 0f}, {0f, 0.5f}};
        float[][]uf= new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}}; float[][]ui= new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[][]uc= new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}}; float[][]uo= new float[][]{{0.2f, 0.1f}, {0.1f, 0.2f}};
        float[]target= new float[]{0.5f, 0.5f};
        Lstm lstm= new Lstm(bf,bi,bc,bo, wf,wi,wc,wo, uf,ui,uc,uo, target);
        lstm.cell.cs.add(new float[]{0f, 0f});
        lstm.cell.hs.add(new float[]{0f, 0f});
        float[]x1= new float[]{1.0f, 0f};
        float[]x2= new float[]{0f, 1.0f};
        float[]x3= new float[]{1.0f, 1.0f};
        lstm.forwarding(0, x1, x2, x3);
    }
}
