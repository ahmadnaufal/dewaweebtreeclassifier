/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dewaweebtreeclassifier;
import java.util.*;
import weka.classifiers.*;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 *
 * @author Aufar
 */
public class Sujeong extends AbstractClassifier {
    Sujeong[] children = null;
    Attribute bestAttr = null;
    Discretize filter = null;
    double value = -1.0;

    public double computeGain(Instances data, Attribute attr) {
        double informationGain = computeEntropy(data);
        Instances[] splitInstances = splitInstancesOnAttribute(data, attr);
        for (Instances instances: splitInstances) {
            informationGain -= ((double) instances.numInstances()/(double) data.numInstances()) * computeEntropy(instances);
        }
        
        return informationGain;
    }

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

    public void buildTree(Instances instances) throws java.lang.Exception {
        if (instances.numAttributes() < 1) {
            throw new Exception("Data instances need to have minimum of 1 attribute.");
        }
        else if (instances.numAttributes() == 1) {
            this.value = instances.meanOrMode(instances.classIndex());
        }
        else {
            Enumeration attrs = instances.enumerateAttributes();
            double informationGain = 0.0;
            while (attrs.hasMoreElements()) {
                Attribute attr = (Attribute)attrs.nextElement();
                double tmpGain = computeGain(instances, attr);
                if (tmpGain > informationGain) {
                    bestAttr = attr;
                    informationGain = tmpGain;
                }
            }
            if (bestAttr != null) {
                double mode = instances.meanOrMode(instances.classIndex());
                Instances[] chunks = splitInstancesOnAttribute(instances, bestAttr);
                children = new Sujeong[chunks.length];
                for (int i = 0; i < chunks.length; ++i) {
                    Instances chunk = chunks[i];
                    Sujeong child = new Sujeong();
                    children[i] = child;
                    if (chunk.numInstances() > 0) child.buildTree(chunk);
                    else child.value = mode;
                }
            }
            else {
                this.value = instances.meanOrMode(instances.classIndex());
            }
        }
    }
    
    @Override
    public void buildClassifier(Instances instances) throws java.lang.Exception {
        filter = new Discretize();
        filter.setInputFormat(instances);
        this.buildTree(Filter.useFilter(instances, filter));
    }
    
   public Instances[] splitInstancesOnAttribute(Instances data, Attribute attr) {
        Instances[] splitInstances = new Instances[attr.numValues()];
        
        for (int i = 0; i < attr.numValues(); i++) {
            splitInstances[i] = new Instances(data, data.numInstances());
        }
        
        Enumeration enumInstance = data.enumerateInstances();
        while (enumInstance.hasMoreElements()) {
            Instance instance = (Instance) enumInstance.nextElement();
            splitInstances[(int) instance.value(attr)].add(instance);
        }
        
        for (int i = 0; i < attr.numValues(); i++) {
            splitInstances[i].compactify();
        }
        
        return splitInstances;
    }

    public double classifyInstance(Instance instance) throws java.lang.Exception {
        if (filter != null) {
            filter.input(instance);
            instance = filter.output();
        }
        if (bestAttr == null) return this.value;
        else {
            return children[(int)instance.value(bestAttr)].classifyInstance(instance);
        }
    }
}
