/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dewaweebtreeclassifier;

import dewaweebtreeclassifier.veranda.VerandaTree;
import java.util.Enumeration;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.classifiers.*;
import weka.core.Attribute;
import weka.core.Instance;
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
     */
    public Veranda() {
        mRoot = new VerandaTree();
    }
    
    /**
     * 
     * @param data 
     * @throws java.lang.Exception 
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (!data.classAttribute().isNominal())
            throw new Exception("The class attribute is not nominal.");
        
        if (!isAllNominalAttributes(data))
            throw new Exception("An attribute has non-nominal value.");
        
        if (isHaveMissingAttributes(data))
            throw new Exception("An instance has missing value(s). ID3 does not support missing values.");
        
        mRoot.buildClassifier(data);
    }
    
    /**
     * 
     * @param instance
     * @return 
     * @throws java.lang.Exception 
     */
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (instance.hasMissingValue())
            throw new Exception("The instance has missing value(s). ID3 does not support missing values.");
        
        return mRoot.classifyInstance(instance);
    }
    
    /**
     * 
     * @param data
     * @return
     */
    public boolean isHaveMissingAttributes(Instances data) {
        Enumeration enumInst = data.enumerateInstances();
        while (enumInst.hasMoreElements()) {
            Instance instance = (Instance) enumInst.nextElement();
            if (instance.hasMissingValue()) {
                return true;
            }
        }
        
        return false;
    }

    /**
     * 
     * @param data
     * @return 
     */
    private boolean isAllNominalAttributes(Instances data) {
        Enumeration enumAttr = data.enumerateAttributes();
        while (enumAttr.hasMoreElements()) {
            Attribute attr = (Attribute) enumAttr.nextElement();
            if (!attr.isNominal()) {
                return false;
            }
        }
        
        return true;
    }
    
}
