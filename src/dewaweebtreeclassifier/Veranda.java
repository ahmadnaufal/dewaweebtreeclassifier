/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dewaweebtreeclassifier;

import dewaweebtreeclassifier.veranda.VerandaTree;
import weka.classifiers.*;
import weka.core.Instances;
/**
 * Veranda is a decision tree classifier model based on ID3 Decision Tree Algorithm
 * @author Ahmad
 */
public class Veranda
        extends AbstractClassifier {
    
    protected VerandaTree mRoot;
    
    /**
     * 
     * @param data 
     */
    @Override
    public void buildClassifier(Instances data) {
        mRoot.buildClassifier(data);
    }
    
}
