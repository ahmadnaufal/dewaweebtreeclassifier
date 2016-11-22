package dewaweebtreeclassifier;

/**
 * Created by Ahmad on 11/22/2016.
 */
public class Node {
    public Node left;
    public Node right;
    public Node parent;
    public int iInstanceLeft;
    public int iInstanceRight;

    public double mLevel = 0;

    public void setLevel(double level) {
        mLevel = level;
    }
}
