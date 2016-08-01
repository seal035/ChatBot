package DL4j.DL4J;

import java.io.IOException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.models.*;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.util.Arrays;  
import java.util.List;

/**
 * Hello world!
 *
 */
public class App 
{
	
    public static void main( String[] args ) throws IOException
    {
    	final int numRows = 28; 
    	final int numColumns = 28; 
    	int seed = 123; 
    	int numSamples = MnistDataFetcher.NUM_EXAMPLES; 
    	int batchSize = 1000; 
    	int iterations = 1; 
    	int listenerFreq = iterations/5;
    	Logger log = LoggerFactory.getLogger(App.class);
    	  log.info("Load data....");
    	    DataSetIterator iter = new MnistDataSetIterator(batchSize,numSamples,true);

    	    log.info("Build model....");
    	    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    	            .seed(seed)
    	            .iterations(iterations)
    	            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    	            .list(10)
    	            .layer(0, new RBM.Builder().nIn(numRows * numColumns).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
    	            .layer(1, new RBM.Builder().nIn(1000).nOut(500).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
    	            .layer(2, new RBM.Builder().nIn(500).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
    	            .layer(3, new RBM.Builder().nIn(250).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
    	            .layer(4, new RBM.Builder().nIn(100).nOut(30).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) 
    	            
    	            //encoding stops
    	            .layer(5, new RBM.Builder().nIn(30).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) 	
    	            
    	            //decoding starts
    	            .layer(6, new RBM.Builder().nIn(100).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
    	            .layer(7, new RBM.Builder().nIn(250).nOut(500).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
    	            .layer(8, new RBM.Builder().nIn(500).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
    	            .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT).nIn(1000).nOut(numRows*numColumns).build())
    	            .pretrain(true).backprop(true)
    	            .build();

    	     MultiLayerNetwork model = new MultiLayerNetwork(conf);
    	     model.init();

    	     model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

    	     log.info("Train model....");
    	     while(iter.hasNext()) {
    	        DataSet next = iter.next();
    	        model.fit(new DataSet(next.getFeatureMatrix(),next.getFeatureMatrix()));
    }
}
}
