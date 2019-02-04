# Deeplearning4j - OnDevice (Mobile) Word2Vec Training 
# Word2Vec Porting On Android Using DeepLearning4j
# Model training takes place on mobile and creates a word2vec model. 
# Once training is done - word2vec.wordnearestsum returns nearest trained wordvector.
# CHOOSE HIGH END DEVICE FOR RUNNING THE CODE - [ Tested Device - Samsung Galaxy S7 ]
# File Folders
# assets/
/vector_data1.txt
/vector_data2.txt
/vector_data3.txt
/vector_data4.txt
# raw/
stopwords.txt
/extended_stopwords.txt
# layout/
activity_main.xml
# main/java/
MainActivity.java
/WordVectorReader.java
/WordVectorSaver.java
/WordVectorTraining.java
# Storage Permission
Manually Enable Permission from App Settings For the First Time or Handle it from code.
# Delete W2V_DATAPATH
Manually delete trained dataset for including new training data or go for uptraining process.
# Running word2vec algorithm on mobile getting ANR in Base64 library due to java and android compatibility
# WordVectorReader and WordVectorSaver, overrides the funtion of library in order to solve the dependencies
# Tried to solve encodeBase64/decodeBase64 compatibility issue between android and java: https://github.com/auth0/java-jwt/issues/131
# Data Set Source : http://www.gutenberg.org/

# If you see below issue try with high end devices

# ISSUE DISCUSSION
https://gitter.im/deeplearning4j/deeplearning4j/archives/2016/09/06  

# ISSUE_1
java.lang.IllegalStateException: You can't fit() model with empty Vocabulary or WeightLookupTable
        at org.deeplearning4j.models.sequencevectors.SequenceVectors.fit(SequenceVectors.java:238)
        at com.example.vijay.ondevice_word2vector.WordVectorTraining.trainW2V(WordVectorTraining.java:174)
        at com.example.vijay.ondevice_word2vector.MainActivity$1.run(MainActivity.java:50)
        at java.lang.Thread.run(Thread.java:764)
		
# ISSUE_2		
2019-02-02 21:17:14.890 30303-30303/? A/DEBUG: pid: 30179, tid: 30284, name: VectorCalculati  
>>> com.example.vijay.ondevice_word2vector <<<
2019-02-02 21:17:14.890 30303-30303/? A/DEBUG: signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x0
2019-02-02 21:17:14.890 30303-30303/? A/DEBUG: Cause: null pointer dereference
2019-02-02 21:17:14.890 30303-30303/? A/DEBUG:     r0 00000000  r1 00000000  r2 00000001  r3 00000000
2019-02-02 21:17:14.890 30303-30303/? A/DEBUG:     r4 00000000  r5 00000064  r6 00000000  r7 dc0212b8
2019-02-02 21:17:14.890 30303-30303/? A/DEBUG:     r8 00000000  r9 00000000  sl 00000064  fp b15f9450
2019-02-02 21:17:14.890 30303-30303/? A/DEBUG:     ip 00000001  sp b15f93d0  lr b21f1800  pc b7f5ca24  cpsr 200e0010

# IDEA BEHIND TRAINING

(a) The whole idea is to train word2vec on device (mobile), mentioned issue above may not appear for high end mobile devices.
(b) Code successfully run for training configuration in code and vector_data1.txt"
(c) With the current vector_data1.txt, desired result is not obtained. Increasing the data causing ANR as mentioned in Issue.
(d) Tune the parameters, try with various mobile devices, clear the caches, free RAM usage during training etc. for obtaining the desired result.
(e) Doing training ondevice can help in securing user privacy.

# Below code will work in high end device
# Change or modify /asset/vector_data[*].txt for domain specific word2vec 
# Tune hyper parameters for best result
                word2Vec[i] = new Word2Vec.Builder()
                        .minWordFrequency(10)  /*observation  : This has major role to play for mentioned ISSUE*/
                        .iterations(1)
                        .layerSize(100)
                        .seed(42)
                        .windowSize(5)
                        .epochs(1)  /*observation  : It will take more time in training*/
                        .batchSize(10) /*Commented*/
                        .stopWords(stopwords) /*observation  : These worked for high end mobile devices*/
                        .stopWords(extendedStopwords) /*observation  : These worked for high end mobile devices*/
                        .iterate(iterator)
                        .tokenizerFactory(tokenizerFactory)
                        .lookupTable(table) /*Commented*/
                        .vocabCache(cache) /*Commented*/
                        .elementsLearningAlgorithm(new SkipGram<VocabWord>())
                        .build();
![alt text] (../master/Deeplearning4j/Screenshot_20190203-131421_OnDevice_Word2Vector.jpg)
