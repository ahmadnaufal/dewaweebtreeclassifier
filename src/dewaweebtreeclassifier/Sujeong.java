/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dewaweebtreeclassifier;
import java.util.*;
import weka.classifiers.*;
import weka.core.*;

/**
 *
 * @author Aufar
 */
public class Sujeong extends AbstractClassifier {
    Sujeong children[] = null;
    double value = 0.0;

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

        double entropy = 0;
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
            Attribute bestAttr = null;
            while (attrs.hasMoreElements()) {
                Attribute attr = (Attribute)attrs.nextElement();
                double tmpGain = computeGain(instances, attr);
                if (tmpGain > informationGain) {
                    bestAttr = attr;
                }
            }
            if (bestAttr != null) {
                double mode = instances.meanOrMode(instances.classIndex());
                int chId = 0;
                Instances[] chunks = splitInstancesOnAttribute(instances, bestAttr);
                children = new Sujeong[chunks.length];
                for (Instances chunk: chunks) {
                    if (chunk.numInstances() > 0) children[chId++].buildTree(chunk);
                    else children[chId++].value = mode;
                }
            }
            else {
                throw new Exception("Information Gain < 0.0");
            }
        }
    }
    public void buildClassifier(Instances instances) throws java.lang.Exception {
        for (Instance inst: instances) {
        }
        this.buildTree(instances);
    }
    public Instances[] splitInstancesOnAttribute(Instances data, Attribute attr) {
        Instances[] splitInstances = new Instances[attr.numValues()];

        Enumeration enumInstance = data.enumerateInstances();
        while (enumInstance.hasMoreElements()) {
            Instance instance = (Instance) enumInstance.nextElement();
            splitInstances[(int) instance.value(attr)].add(instance);
        }

        return splitInstances;
    }

    public double classifyInstance(Instance instances) throws java.lang.Exception {
        return 0.0;
    }
}
