/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dewaweebtreeclassifier;

import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Scanner;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.supervised.instance.Resample;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Ahmad
 */
public class DewaWeebTreeClassifier {

    private static Instances data;

    public static void printMain() {
        String dialog[] = {
            "Hai, apa yang ingin kamu lakukan?",
            "1. Load File",
            "2. Filter",
            "3. Remove Attribute",
            "4. Display Current Data",
            "5. Training",
            "6. Classify",};
        for (String line : dialog) {
            System.out.println(line);
        }
    }

    enum Menu {
        NOOP,
        LOAD_FILE,
        FILTER,
        REMOVE,
        DISPLAY_DATA,
        TRAINING,
        CLASSIFY
    }

    static Menu mainMenu[] = {
        Menu.NOOP,
        Menu.LOAD_FILE,
        Menu.FILTER,
        Menu.REMOVE,
        Menu.DISPLAY_DATA,
        Menu.TRAINING,
        Menu.CLASSIFY
    };

    /**
     * 
     * @param path the .arff or .csv file path
     * @throws Exception 
     */
    private void loadFile(String path) throws Exception {
        // Load data
        DataSource source = new DataSource(path);
        data = source.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
    }
    
    /**
     * resample instances from the whole dataset
     * @return new dataset after resampling
     * @throws Exception 
     */
    private Instances resampleInstances() throws Exception {
        Resample sampler = new Resample();
        sampler.setInputFormat(data);
        
        Instances newData = Resample.useFilter(data, sampler);
        return newData;
    }
    
    /**
     * 
     * @param idxAttrs the list of indexes of the attributes to be removed (from 0)
     * @param invertSelection if true, attributes in idxAttrs will be kept, instead of being removed
     * @return new dataset after attribute removal
     */
    private Instances removeAttribute(int[] idxAttrs, boolean invertSelection) throws Exception {
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(idxAttrs);
        remove.setInvertSelection(invertSelection);
        remove.setInputFormat(data);
        
        Instances newData = Filter.useFilter(data, remove);
        return newData;
    }

    public void run() {
        while (true) {
            Scanner sc = new Scanner(System.in);
            int choiceNum = sc.nextInt();
            Menu chosenMenu = mainMenu[choiceNum];

            try {
                switch (chosenMenu) {
                    case LOAD_FILE:
                        System.out.print("Please input file path: ");
                        String filepath = sc.nextLine();
                        this.loadFile(filepath);
                        break;
                    case FILTER:
                        // resample case: Get the dataset and re-sample it to get a new dataset
                        data = this.resampleInstances();
                        break;
                    case REMOVE:
                        // remove attributes
                        System.out.print("Please input the attribute index to be removed: ");
                        String idxStr = sc.nextLine();
                        String[] listIdxStr = idxStr.split("\\s+");
                        int[] listIdx = new int[listIdxStr.length];
                        for (int i = 0; i < listIdx.length; i++) {
                            listIdx[i] = Integer.parseInt(listIdxStr[i]);
                        }
                        System.out.print("Invert selection? (Y/N)");
                        String invStr = sc.nextLine();
                        boolean isInv = false;
                        if (invStr.toLowerCase().compareTo("y") == 0)
                            isInv = true;
                        
                        data = this.removeAttribute(listIdx, isInv);
                        break;
                    case DISPLAY_DATA:
                        // Train data
                        break;  
                    case TRAINING:
                        // Train data
                        break;   
                    case CLASSIFY:
                        // Copy data
                        Instances classified = new Instances(data);
                        // Classify data
                        for (int i = 0; i < data.numInstances(); i++) {
                            double classifiedLabel = tree.classifyInstance(data.instance(i));
                            classified.instance(i).setClassValue(classifiedLabel);
                        }
                        break;
                }
            } catch (Exception ex) {
                System.out.println(ex.getMessage());
            }

            // Remove Attribute
            // Resample
            // TODO: Pecah data jadi training set dan testing set dengan percentage split.
            // Build Classifier
            String[] options = new String[1];
            options[0] = "-U";
            J48 tree = new J48();
            tree.setOptions(options);
            // Testing model given test set
            // 10-fold cross validation, percentage split
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(tree, data, 10, new Random(1));
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            // Save / Load model using serialization
            // Save
            weka.core.SerializationHelper.write("/some/where/j48.model", tree);
            // Load
            Classifier cls = (Classifier) weka.core.SerializationHelper.read("/some/where/j48.model");
            // Using model to classify unseen data
            Instances unclassified = new Instances(new BufferedReader(new FileReader("file.arff")));
            unclassified.setClassIndex(unclassified(unclassified.numAttributes() - 1));
            // Copy data
            Instances classified = new Instances(unclassified);
            // Classify data
            for (int i = 0; i < unclassified.numInstances(); i++) {
                double classifiedLabel = tree.classifyInstance(unclassified.instance(i));
                classified.instance(i).setClassValue(classifiedLabel);
            }   // Save data
            BufferedWriter writer = new BufferedWriter(
                    new FileWriter("classified.arff"));
            writer.write(labeled.toString());
            writer.newLine();
            writer.flush();
            writer.close();

        }
    }

    public static void main(String[] args) {
        try {
            DewaWeebTreeClassifier app = new DewaWeebTreeClassifier();
            app.run();
        } catch (Exception ex) {
            Logger.getLogger(DewaWeebTreeClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

}
