package com.example.vijay.ondevice_word2vector;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.widget.Toast;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class WordVectorTraining {

    private static String TAG = "WordVectorTraining";
    private static String DATA_PATH = Environment.getExternalStorageDirectory().toString()+"/W2V_DATAPATH/";
    private Context context;
    private WordVectorSaver wordVectorSaver;
    private WordVectorReader wordVectorReader;
    private final List<String> stopwords = new ArrayList<String>();
    private final List<String> extendedStopwords = new ArrayList<String>();

    private InputStream [] in_vectorStream ;
    private static File[]datafile;
    private Word2Vec []word2Vec = null;
//     private static String []VectorModelFile = {"vector_data1.txt" ,
//                                                "vector_data2.txt" ,
//                                                "vector_data3.txt",
//                                                "vector_data4.txt",
//                                                "vector_data5.txt",
//                                                "vector_data6.txt"
//     };

    /*Keep this in W2V_DATAPATH for direct reading the files - worked for Galaxy S7 device*/
    private static String []VectorModelFile = {"glove.6B.50d"};
    
    private static final String MSG_KEY = "training";

    public WordVectorTraining(Context context) throws IOException {

        this.context = context;

        wordVectorSaver = new WordVectorSaver(context);
        wordVectorReader = new WordVectorReader(context);

        AssetManager assetManager = context.getAssets();

        /*Read # of data files from assest manager*/
        try{
            in_vectorStream = new InputStream[VectorModelFile.length];
            for(int i = 0 ; i < VectorModelFile.length ; i++){
                in_vectorStream[i] = assetManager.open(VectorModelFile[i]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        /*Create Storage Directory for dumping # of model files generated from # of data files*/
        if(!DATA_PATH.endsWith(File.separator)){
            DATA_PATH += File.separator;
        }
        File datapathFiles = new File(DATA_PATH);
        if(!datapathFiles.exists()){
            datapathFiles.mkdir();
        }

        datafile = new File[VectorModelFile.length];
        boolean status = false;
        for(int i = 0; i<VectorModelFile.length ; i++){
            datafile[i] = new File(DATA_PATH+VectorModelFile[i]);
            if(!datafile[i].exists()){ /*Create # of model files*/
                datafile[i].createNewFile();
                status = true;
            }
        }
        if(status){
            wordVectorSaver.resetSharedpreferences();
        }else{
            wordVectorSaver.setSharedpreferences(); //Add this for direct reading of glove.6B.50d
        }

        /*Load Stopwords*/
        InputStream stop = context.getResources().openRawResource(R.raw.stopwords);
        InputStream exstop = context.getResources().openRawResource(R.raw.extended_stopwords);

        BufferedReader br = new BufferedReader(new InputStreamReader(stop));
        String line;
        while((line = br.readLine()) != null){
            stopwords.add(line);
        }
        br.close();
        br = new BufferedReader(new InputStreamReader(exstop));
        while((line = br.readLine()) != null){
            extendedStopwords.add(line);
        }
        br.close();
    }

    private final Handler mHandler = new Handler(){
        public void handleMessage(Message msg){
            Bundle bundle = msg.getData();
            String string = bundle.getString(MSG_KEY);
            Toast toast = Toast.makeText(context,string,Toast.LENGTH_LONG);
        }
    };

    private final Runnable mMessageSender = new Runnable() {
        @Override
        public void run() {
            Message msg = mHandler.obtainMessage();
            Bundle bundle = new Bundle();
            bundle.putString(MSG_KEY,"Training In Progress");
            msg.setData(bundle);
            mHandler.sendMessage(msg);
        }
    };
    
    public void trainW2V(){
        SentenceIterator iterator = null;
        TokenizerFactory tokenizerFactory = null;
        VocabCache<VocabWord> cache = null;
        WeightLookupTable<VocabWord> table = null;
        word2Vec  = new Word2Vec[in_vectorStream.length];

        if(wordVectorSaver.getSavedModelState() == false){

            new Thread(mMessageSender).start();
            for(int i = 0 ; i < in_vectorStream.length ; i++) { /*Train model iteratively with dataset[]*/

                iterator = new BasicLineIterator(in_vectorStream[i]);
                tokenizerFactory = new DefaultTokenizerFactory();
                tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
                Log.d(TAG, "Building Model");

                // manual creation of VocabCache and WeightLookupTable usually isn't necessary
                // but in this case we'll need them
//                cache = new AbstractCache<>();
//                table = new InMemoryLookupTable.Builder<VocabWord>()
//                        .vectorLength(100)
//                        .useAdaGrad(false)
//                        .cache(cache).build();

                word2Vec[i] = new Word2Vec.Builder()
                        .minWordFrequency(10)
                        .iterations(1)
                        .layerSize(100)
                        .seed(42)
                        .windowSize(5)
                        .epochs(1)
//                        .batchSize(10)
//                        .stopWords(stopwords)
//                        .stopWords(extendedStopwords)
                        .iterate(iterator)
                        .tokenizerFactory(tokenizerFactory)
//                        .lookupTable(table)
//                        .vocabCache(cache)
                        .elementsLearningAlgorithm(new SkipGram<VocabWord>())
                        .build();

                word2Vec[i].fit();

                wordVectorSaver.writeWord2VecModel(word2Vec[i], datafile[i]);
            }
            wordVectorSaver.setSharedpreferences();
        }else{ /*Only read model */
            for(int i = 0 ; i < datafile.length ; i++){

                word2Vec[i] = wordVectorReader.readWord2VecModel(datafile[i]);

                /*Uptraining Process*/
//                iterator = new BasicLineIterator(in_vectorStream[i]);
//                tokenizerFactory = new DefaultTokenizerFactory();
//                tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
//                word2Vec[i].setTokenizerFactory(tokenizerFactory);
//                word2Vec[i].setSentenceIterator(iterator);
            }
        }
    }

    public Word2Vec[] getW2VInstance(){
        if(word2Vec.length > 0){
            return word2Vec;
        }
        return null;
    }

}
