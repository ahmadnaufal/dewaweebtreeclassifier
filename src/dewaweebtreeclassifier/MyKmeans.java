package dewaweebtreeclassifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Vector;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

/**
 * <!-- globalinfo-start --> K-Means clustering class. <!-- globalinfo-end -->
 *
 * <!-- options-start --> Valid options are:
 *
 * <pre>
 * -N
 *  number of clusters
 * </pre>
 *
 * <pre>
 * -D
 * If set, classifier is run in debug mode and may output additional info to the console.
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Aufar Gilbran
 * @author Ahmad Naufal Farhan
 * @author Adin Baskoro Pratomo
 * @version $Revision$
 */
public class MyKmeans
        extends
        weka.clusterers.AbstractClusterer
        implements
        TechnicalInformationHandler {

    Instances instances;
    int numOfClusters = 3;
    Instances centroids;
    ArrayList<Instance> centroidList;
    ArrayList< Instances> clusters;
    DistanceFunction distanceFunction;

    public int getNumOfClusters() {
        return numOfClusters;
    }
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capability.NO_CLASS);
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);

        return result;
    }

    @Override
    public String doNotCheckCapabilitiesTipText() {
        return super.doNotCheckCapabilitiesTipText();
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        // Check data
        instances = data;
        int numInstances = instances.numInstances();
        if (numInstances == 0) {
            return;
        }
        // Use Euclidean as distance function
        distanceFunction = new EuclideanDistance();
        distanceFunction.setInstances(instances);

        // Initialize centroid
        centroids = new Instances(instances, numOfClusters);
        Random rand = new Random();
        int prevIdx = -1;
        for (int i = 0; i < numOfClusters; i++) {
            Instance centroid;
            int clusterIdx = rand.nextInt(numOfClusters);
            while (clusterIdx == prevIdx) {
                clusterIdx = rand.nextInt(numOfClusters);
            }
            centroid = new DenseInstance(instances.instance(clusterIdx));
            prevIdx = clusterIdx;
            centroids.add(centroid);
        }

        // Cluster all instances. Iterate until converges or max iteration limit exceeded
        int clusterAssignments[] = new int[instances.numInstances()];
        int[] oldAssignments = new int[instances.numInstances()];
        boolean converged = false;
        int it = 0;
        while (!converged || it == 1000) {
            // Cluster all instances.
            System.arraycopy(clusterAssignments, 0, oldAssignments, 0, clusterAssignments.length);
            for (int j = 0; j < instances.numInstances(); j++) {
                Double best = Double.MAX_VALUE;
                int ctrIdx = -1;
                // Check each clusters
                for (int i = 0; i < centroids.numInstances(); i++) {
                    Double temp = distanceFunction.distance(centroids.instance(i), instances.instance(j));
//                    System.out.println("Instance " + String.valueOf(j));
//                    System.out.println("Distance to cluster" + String.valueOf(i));
//                    System.out.println(temp);
                    if (temp <= best) {
                        best = temp;
                        ctrIdx = i;
                    }
                }
                clusterAssignments[j] = ctrIdx;
            }
            // Calculate new centroid
            Instances newCentroids = new Instances(centroids, numOfClusters);
            for (int i = 0; i < centroids.numInstances(); i++) {
                Instances temp = new Instances(instances);
                double[] values = new double[temp.numAttributes()];
                for (int j = 0; j < instances.numInstances(); j++) {
                    if (clusterAssignments[j] == i) {
                        temp.add(instances.instance(i));
                    }
                }
                // Calculate means
                for (int j = 0; j < temp.numAttributes(); j++) {
                    values[j] = temp.meanOrMode(j);
                }
//                System.out.println(Arrays.toString(values));
                DenseInstance nc = new DenseInstance(1.0, values);
                newCentroids.add(nc);
            }
            centroids = newCentroids;
            if (Arrays.equals(oldAssignments, clusterAssignments)) {
                converged = true;
            }
            else {
                it++;
            }
        }

        // Print clustering result
        for (int i = 0; i < centroids.numInstances(); i++) {
            System.out.println("Cluster " + String.valueOf(i) + ": ");
            for (int j = 0; j < clusterAssignments.length; j++) {
                if (clusterAssignments[j] == i) {
                    System.out.print(String.valueOf(j) + " ");
                }
            }
            System.out.println("");
        }
    }

    @Override
    public int clusterInstance(Instance instance) throws Exception {
        double best = Double.POSITIVE_INFINITY;
        int cluster = 0;
        for (int i = 0; i < centroids.numInstances(); i++) {
            Instance ctr = centroids.instance(i);
            double temp = distanceFunction.distance(ctr, instance);
            if (temp < best) {
                best = temp;
                cluster = i;
            }
        }
        return cluster;
    }

    @Override
    public int numberOfClusters() throws Exception {
        return Math.min(numOfClusters, instances.numInstances());
    }

    @Override
    public String[] getOptions() {
        Vector<String> options = new Vector();
        options.add("-N");
        options.add("" + getNumOfClusters());
        return options.toArray(new String[0]);
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String optionString = Utils.getOption('N', options);
        if (optionString.length() != 0) {
            numOfClusters = new Integer(optionString);
        } else {
            numOfClusters = 2;
        }
        Utils.checkForRemainingOptions(options);
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
