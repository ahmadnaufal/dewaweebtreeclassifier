package dewaweebtreeclassifier;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import weka.clusterers.AbstractClusterer;
import weka.core.EuclideanDistance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.TreeSet;
import weka.core.DistanceFunction;

/**
 * Created by Ahmad on 11/20/2016.
 */
public class MyAgnes extends AbstractClusterer {

    private Instances mInstances;
    private int mNumClusters;
    DistanceFunction mDistanceFunction;
    Node[] mTree;

    public MyAgnes() {
        mNumClusters = 2;
    }

    public MyAgnes(int clusters) {
        mNumClusters = clusters;
    }
    
    class DSU {
        public ArrayList<Integer> par;
        DSU(int n) {
            this.par = new ArrayList<>();
            for (int i = 0; i < n; ++i) {
                par.add(i);
            }
        }
        
        public int getParent(int v) {
            if (v == par.get(v)) return v;
            else {
                par.set(v, this.getParent(par.get(v)));
                return par.get(v);
            }
        }
        
        public boolean sameset(int u, int v) {
            int pu = this.getParent(u);
            int pv = this.getParent(v);
            return pu == pv;
        }
        
        public void join(int u, int v) {
            int pu = this.getParent(u);
            int pv = this.getParent(v);
            if (pv > pu) par.set(pv, pu);
            else par.set(pu, pv);
        }
    }

    @Override
    public void buildClusterer(Instances instances) throws Exception {
        mInstances = instances;
        mDistanceFunction = new EuclideanDistance(instances);
        if (mInstances.numInstances() < 1) {
            System.out.println("No input data from the instances.");
            return;
        }

        int nClusters = mInstances.numInstances();
        ArrayList<Integer> [] dataClusters = new ArrayList[nClusters];
        for (int i = 0; i < nClusters; ++i) {
            // assign all instance index to all clusters
            dataClusters[i] = new ArrayList<>();
            dataClusters[i].add(i);
        }

        // join nodes based on euclidean distances
        mTree = new Node[nClusters];
        double[][] initDistances = initDistancesPerInstance(nClusters, dataClusters);
        double[][] distances = initDistances.clone();
                
        // start merging clusters
        DSU ds = new DSU(mInstances.numInstances());
        while (mNumClusters < nClusters) {
            int iMin = -1, jMin = -1;
            double minDistance = Double.MAX_VALUE;
            
            for (int i = 0; i < mInstances.numInstances(); ++i) {
                if (dataClusters[i].size() > 0) {
                    for (int j = i+1; j < mInstances.numInstances(); ++j) {
                        if (initDistances[i][j] < minDistance && !ds.sameset(i, j)) {
                            minDistance = initDistances[i][j];
                            iMin = i;
                            jMin = j;
                        }
                    }
                }
            } 
            
            ds.join(iMin, jMin);
            

            nClusters--;
        }
        
        TreeSet<Integer> ts = new TreeSet<>();
        for (int i = 0; i < mInstances.numInstances(); ++i) {
            ts.add(ds.getParent(i));
        }
        int clusterId = 0;
        for (Integer p: ts) {
            clusterId++;
            System.out.print("Cluster " + clusterId + ": ");
            boolean first = true;
            int cnt = 0;
            for (int i = 0; i < mInstances.numInstances(); ++i)
                if (ds.getParent(i) == p) {
                    if (!first) System.out.print(", ");
                    System.out.print(i);
                    first = false;
                    ++cnt;
                }
            System.out.println("");
            System.out.println("Size: " + cnt);
        }
    }

    private double[][] initDistancesPerInstance(int numClusters, ArrayList<Integer>[] dataClusters) {
        double[][] distanceMatrix = new double[numClusters][numClusters];
        for (int i = 0; i < numClusters; ++i) {
            distanceMatrix[i][i] = 0;
            for (int j = i+1; j < numClusters; ++j) {
                distanceMatrix[j][i] = distanceMatrix[i][j] = getInitClusterDistance(dataClusters[i], dataClusters[j]);
                // distanceMatrix[j][i] = distanceMatrix[i][j];
            }
        }

        return distanceMatrix;
    }

    private double getClusterDistance(double [][] initDistance, ArrayList<Integer> cluster1, ArrayList<Integer> cluster2) {
        // we are using single link
        // find the minimum link
        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < cluster1.size(); ++i) {
            for (int j = 0; j < cluster2.size(); ++j) {
                if (initDistance[i][j] < minDistance)
                    minDistance = initDistance[i][j];
            }
        }

        return minDistance;
    }

    private double getInitClusterDistance(ArrayList<Integer> cluster1, ArrayList<Integer> cluster2) {
//        assert (mDistanceFunction != null);
//        assert (mInstances != null);
//        assert (cluster1 != null);
//        assert (cluster2 != null);
//        System.out.println(mInstances.instance(cluster1.get(0)));
//        System.out.println(mInstances.instance(cluster2.get(0)));
        return mDistanceFunction.distance(mInstances.instance(cluster1.get(0)), mInstances.instance(cluster2.get(0)));
    }

    private void mergeClusters(int i1, int i2, double distance, ArrayList<Integer>[] dataClusters, Node[] tree) {
        dataClusters[i1].addAll(dataClusters[i2]);
        dataClusters[i2].clear();

        Node node = new Node();
        if (tree[i1] == null) {
            node.iInstanceLeft = i1;
        } else {
            node.left = tree[i1];
            tree[i1].parent = node;
        }

        if (tree[i2] == null) {
            node.iInstanceRight = i2;
        } else {
            node.right = tree[i2];
            tree[i2].parent = node;
        }

        node.setLevel(distance);
    }

    @Override
    public int numberOfClusters() throws Exception {
        return Math.min(mInstances.numInstances(), mNumClusters);
    }
}
