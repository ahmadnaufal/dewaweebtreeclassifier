/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dewaweebtreeclassifier;

import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.supervised.instance.Resample;
import weka.classifiers.trees.J48;

/**
 *
 * @author Ahmad
 */
public class DewaWeebTreeClassifier {


    public static void main(String[] args) {
        // Load data
        DataSource source = new DataSource(args[0]);
        Instances test, train;
        Instances data = source.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        // Remove Attribute

        // Resample
        // TODO: Pecah data jadi training set dan testing set dengan percentage split.
        

        // Build Classifier
        String[] options = new String[1];
        options[0] = "-U";
        J48 tree = new J48();
        tree.setOptions(options);
        tree.buildClassifier(training);

        // Testing model given test set
        // 10-fold cross validation, percentage split
        Evaluation eval = new Evaluation(data);
        eval.crossValidationModel(tree, data, 10, new Random(1));
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));

        // Save / Load model using serialization
        // Save
        weka.core.SerializationHelper.write("/some/where/j48.model", cls);

        // Load
        Classifier cls = (Classifier) weka.core.SerializationHelper.read("/some/where/j48.model");

        // Using model to classify unseen data
        Instances unclassified = new Instances(new BufferedReader(new FileReader("file.arff")));
        unclassified.setClassIndex(unclassified(unclassified.numAttributes() - 1));

        // Copy data
        Instances classified = new Instances(unclassified);
        
        // Classify data
        for (int i=0;i<unclassified.numInstances();i++){
            double classifiedLabel = tree.classifyInstance(unclassified.instance(i));
            classified.instance(i).setClassValue(classifiedLabel);
        }

        // Save data
        BufferedWriter writer = new BufferedWriter(
                new FileWriter("classified.arff"));
        writer.write(labeled.toString());
        writer.newLine();
        writer.flush();
        writer.close();
    }

}
