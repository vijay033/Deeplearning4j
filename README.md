# Deeplearning4j - OnDevice (Mobile) Word2Vec Training 
# Word2Vec Porting On Android Using DeepLearning4j
# Model training takes place on mobile and creates a word2vec model. 
# Once training is done - word2vec.wordnearestsum returns nearest trained wordvector.
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
# Running word2vec algorithm on mobile getting ANR in Base64 library due to java and android compatibility
# WordVectorReader and WordVectorSaver, overrides the funtion of library in order to solve the dependencies
# Tried to solve encodeBase64/decodeBase64 compatibility issue between android and jave: auth0/java-jwt#131
# Data Set Source : http://www.gutenberg.org/
