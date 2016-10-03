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
        CLASSIFY
    }
    
    static Menu mainMenu[] = { 
        Menu.NOOP, 
        Menu.LOAD_FILE, 
        Menu.FILTER, 
        Menu.CLASSIFY
    };
    
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
    
    public void run() {
        while (true) {
            Scanner sc = new Scanner(System.in);
            int choiceNum = sc.nextInt();
            Menu chosenMenu = mainMenu[choiceNum];

            switch (chosenMenu) {
                case LOAD_FILE:
                    System.out.print("Please input file path: ");
                    String filepath = sc.nextLine();
                    try {
                        this.loadFile(filepath);
                    }
                    catch (Exception ex) {
                        System.out.println(ex.getMessage());
                    }
                    break;
                case FILTER:
                    Resample sampler = new Resample();
                    sampler.setInputFormat(data);
                    break;
                case CLASSIFY:
                    break;
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
