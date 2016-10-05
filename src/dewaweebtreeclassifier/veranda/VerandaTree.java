/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dewaweebtreeclassifier.veranda;

import java.util.Enumeration;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Ahmad
 */
public class VerandaTree {
    
    protected VerandaTree[] mChild;
    protected Attribute mSplitAttribute;
    protected double mClassValue;
    protected int[] mClassDistribution;
    
    /**
     * The main constructor
     */
    public VerandaTree() {
        mSplitAttribute = null;
    }
    
    /**
     * 
     * @param data 
     */
    public void buildClassifier(Instances data) {
        // remove all instance with missing class value
        data.deleteWithMissingClass();
        
        buildTree(data);
    }
    
    /**
     * 
     * @param data 
     */
    public void buildTree(Instances data) {
        // exit if there is no data left in the dataset
        if (data.numInstances() == 0) {
            mChild = null;
            return;
        }
        
        double[] informationGains = new double[data.numAttributes()];
        Enumeration enumAttrs = data.enumerateAttributes();
        while (enumAttrs.hasMoreElements()) {
            Attribute attr = (Attribute) enumAttrs.nextElement();
            informationGains[attr.index()] = computeGain(data, attr);
        }
        int maxIdx = Utils.maxIndex(informationGains);
        
        if (Utils.eq(informationGains[maxIdx], 0)) {
            mClassDistribution = new int[data.numClasses()];
            Enumeration enumInst = data.enumerateInstances();
            while (enumInst.hasMoreElements()) {
                Instance instance = (Instance) enumInst.nextElement();
                mClassDistribution[(int) instance.classValue()]++;
            }
            mClassValue = Utils.maxIndex(mClassDistribution);
        } else {
            mSplitAttribute = data.attribute(maxIdx);
            Instances[] splitInstances = splitInstancesOnAttribute(data, mSplitAttribute);
            mChild = new VerandaTree[mSplitAttribute.numValues()];
            for (int i = 0; i < mChild.length; i++) {
                mChild[i].buildTree(splitInstances[i]);
            }
        }        
    }
    
    /**
     * 
     * @param data
     * @param attr
     * @return 
     */
    public double computeGain(Instances data, Attribute attr) {
        double informationGain = computeEntropy(data);
        Instances[] splitInstances = splitInstancesOnAttribute(data, attr);
        for (Instances instances: splitInstances) {
            informationGain -= ((double) instances.numInstances()/(double) data.numInstances()) * computeEntropy(instances);
        }
        
        return informationGain;
    }
    
    /**
     * 
     * @param data 
     * @return  
     */
    public double computeEntropy(Instances data) {
        double[] nClass = new double[data.numClasses()];
        Enumeration enumInstance = data.enumerateInstances();
        while (enumInstance.hasMoreElements()) {
            Instance instance = (Instance) enumInstance.nextElement();
            nClass[(int) instance.classValue()]++;
        }
        
        double entropy = 0.0;
        for (int i = 0; i < data.numClasses(); i++) {
            if (nClass[i] > 0) {
                double ratio = nClass[i]/data.numInstances();
                entropy -= (ratio * Utils.log2(ratio));
            }
        }
        
        return entropy;
    }
    
    /**
     * 
     * @param data
     * @param attr
     * @return 
     */
    public Instances[] splitInstancesOnAttribute(Instances data, Attribute attr) {
        Instances[] splitInstances = new Instances[attr.numValues()];
        
        Enumeration enumInstance = data.enumerateInstances();
        while (enumInstance.hasMoreElements()) {
            Instance instance = (Instance) enumInstance.nextElement();
            splitInstances[(int) instance.value(attr)].add(instance);
        }
        
        return splitInstances;
    }
}
