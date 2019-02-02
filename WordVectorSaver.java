package com.example.vijay.ondevice_word2vector;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Base64;
import android.util.Log;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.output.CloseShieldOutputStream;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.util.DL4JFileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import lombok.NonNull;

public class WordVectorSaver {

    Context context;
    public boolean savedModelflag = false;
    SharedPreferences sharedpreferences;
    SharedPreferences.Editor sharedpreferencesEditor;
    public static String TAG = "WordVectorSaver";

    private static final int MAX_SIZE = 50;
    private static final String WHITESPACE_REPLACEMENT = "_Az92_";

    public WordVectorSaver(Context context){
        this.context = context;
    }

    public void setSharedpreferences(){
        sharedpreferences = context.getSharedPreferences("MyPreferences", Context.MODE_PRIVATE);
        sharedpreferencesEditor = sharedpreferences.edit();
        savedModelflag = true;
        sharedpreferencesEditor.putBoolean("ModelSavedState",true);
        sharedpreferencesEditor.commit();
    }

    public boolean getSavedModelState() {
        boolean flag = false;
        sharedpreferences = context.getSharedPreferences("MyPreferences",Context.MODE_PRIVATE);
        flag = sharedpreferences.getBoolean("ModelSavedState",false);
        return flag;
    }

    public void resetSharedpreferences(){
        sharedpreferences = context.getSharedPreferences("MyPreferences", Context.MODE_PRIVATE);
        sharedpreferencesEditor = sharedpreferences.edit();
        savedModelflag = false;
        sharedpreferencesEditor.putBoolean("ModelSavedState",false);
        sharedpreferencesEditor.commit();
    }

    public static String encodeB64(String word) {
        try {
            return "B64:" + Base64.encodeToString(word.getBytes("UTF-8"),0).replaceAll("(\r|\n)", "");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static String decodeB64(String word) {
        if (word.startsWith("B64:")) {
            String arp = word.replaceFirst("B64:", "");
            try {
                return new String(Base64.decode(arp,0), "UTF-8");
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        } else
            return word;
    }


    /**
     * This method writes word vectors to the given path.
     * Please note: this method doesn't load whole vocab/lookupTable into memory, so it's able to process large vocabularies served over network.
     *
     * @param lookupTable
     * @param path
     * @param <T>
     */
    public static <T extends SequenceElement> void writeWordVectors(WeightLookupTable<T> lookupTable, String path)
            throws IOException {
        try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(path))) {
            writeWordVectors(lookupTable, bos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method writes word vectors to the given file.
     * Please note: this method doesn't load whole vocab/lookupTable into memory, so it's able to process large vocabularies served over network.
     *
     * @param lookupTable
     * @param file
     * @param <T>
     */
    public static <T extends SequenceElement> void writeWordVectors(WeightLookupTable<T> lookupTable, File file)
            throws IOException {
        try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(file))) {
            writeWordVectors(lookupTable, bos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method writes word vectors to the given OutputStream.
     * Please note: this method doesn't load whole vocab/lookupTable into memory, so it's able to process large vocabularies served over network.
     *
     * @param lookupTable
     * @param stream
     * @param <T>
     * @throws IOException
     */
    public static <T extends SequenceElement> void writeWordVectors(WeightLookupTable<T> lookupTable,
                                                                    OutputStream stream) throws IOException {
        VocabCache<T> vocabCache = lookupTable.getVocabCache();

        try (PrintWriter writer = new PrintWriter(new OutputStreamWriter(stream, StandardCharsets.UTF_8))) {
            // saving header as "NUM_WORDS VECTOR_SIZE NUM_DOCS"
            String str = vocabCache.numWords() + " " + lookupTable.layerSize() + " " + vocabCache.totalNumberOfDocs();
            Log.d(TAG,"Saving header: {}"+str);
            writer.println(str);

            // saving vocab content
            int num = vocabCache.numWords();
            for (int x = 0; x < num; x++) {
                T element = vocabCache.elementAtIndex(x);

                StringBuilder builder = new StringBuilder();

                String l = element.getLabel();
                builder.append(encodeB64(l)).append(" ");
                INDArray vec = lookupTable.vector(element.getLabel());
                for (int i = 0; i < vec.length(); i++) {
                    builder.append(vec.getDouble(i));
                    if (i < vec.length() - 1)
                        builder.append(" ");
                }
                writer.println(builder.toString());
            }
        }
    }


    /**
     * This method saves GloVe model to the given output stream.
     *
     * @param vectors GloVe model to be saved
     * @param file path where model should be saved to
     */
    public static void writeWordVectors(@NonNull Glove vectors, @NonNull File file) {
        try (BufferedOutputStream fos = new BufferedOutputStream(new FileOutputStream(file))) {
            writeWordVectors(vectors, fos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method saves GloVe model to the given output stream.
     *
     * @param vectors GloVe model to be saved
     * @param path path where model should be saved to
     */
    public static void writeWordVectors(@NonNull Glove vectors, @NonNull String path) {
        try (BufferedOutputStream fos = new BufferedOutputStream(new FileOutputStream(path))) {
            writeWordVectors(vectors, fos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method saves GloVe model to the given OutputStream
     *
     * @param vectors GloVe model to be saved
     * @param stream OutputStream where model should be saved to
     */
    public static void writeWordVectors(@NonNull Glove vectors, @NonNull OutputStream stream) {
        try {
            writeWordVectors(vectors.lookupTable(), stream);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method saves paragraph vectors to the given output stream.
     *
     * @param vectors
     * @param stream
     */
    @Deprecated
    public static void writeWordVectors(ParagraphVectors vectors, OutputStream stream) {
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(stream, StandardCharsets.UTF_8))) {
            /*
            This method acts similary to w2v csv serialization, except of additional tag for labels
             */

            VocabCache<VocabWord> vocabCache = vectors.getVocab();
            for (VocabWord word : vocabCache.vocabWords()) {
                StringBuilder builder = new StringBuilder();

                builder.append(word.isLabel() ? "L" : "E").append(" ");
                builder.append(word.getLabel().replaceAll(" ", WHITESPACE_REPLACEMENT)).append(" ");

                INDArray vector = vectors.getWordVectorMatrix(word.getLabel());
                for (int j = 0; j < vector.length(); j++) {
                    builder.append(vector.getDouble(j));
                    if (j < vector.length() - 1) {
                        builder.append(" ");
                    }
                }

                writer.write(builder.append("\n").toString());
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Writes the word vectors to the given path. Note that this assumes an in memory cache
     *
     * @param lookupTable
     * @param cache
     *
     * @param path
     *            the path to write
     * @throws IOException
     * @deprecated Use {@link #writeWord2VecModel(Word2Vec, File)} instead
     */
    @Deprecated
    public static void writeWordVectors(InMemoryLookupTable lookupTable, InMemoryLookupCache cache, String path)
            throws IOException {
        try (BufferedWriter write = new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(path, false), StandardCharsets.UTF_8))) {
            for (int i = 0; i < lookupTable.getSyn0().rows(); i++) {
                String word = cache.wordAtIndex(i);
                if (word == null) {
                    continue;
                }
                StringBuilder sb = new StringBuilder();
                sb.append(word.replaceAll(" ", WHITESPACE_REPLACEMENT));
                sb.append(" ");
                INDArray wordVector = lookupTable.vector(word);
                for (int j = 0; j < wordVector.length(); j++) {
                    sb.append(wordVector.getDouble(j));
                    if (j < wordVector.length() - 1) {
                        sb.append(" ");
                    }
                }
                sb.append("\n");
                write.write(sb.toString());

            }
        }
    }

    /**
     * This method saves paragraph vectors to the given file.
     *
     * @param vectors
     * @param path
     */
    @Deprecated
    public static void writeWordVectors(@NonNull ParagraphVectors vectors, @NonNull File path) {
        try (FileOutputStream fos = new FileOutputStream(path)) {
            writeWordVectors(vectors, fos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * This method saves paragraph vectors to the given path.
     *
     * @param vectors
     * @param path
     */
    @Deprecated
    public static void writeWordVectors(@NonNull ParagraphVectors vectors, @NonNull String path) {
        try (FileOutputStream fos = new FileOutputStream(path)) {
            writeWordVectors(vectors, fos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method saves ParagraphVectors model into compressed zip file
     *
     * @param file
     */
    public static void writeParagraphVectors(ParagraphVectors vectors, File file) {
        try (FileOutputStream fos = new FileOutputStream(file);
             BufferedOutputStream stream = new BufferedOutputStream(fos)) {
            writeParagraphVectors(vectors, stream);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method saves ParagraphVectors model into compressed zip file located at path
     *
     * @param path
     */
    public static void writeParagraphVectors(ParagraphVectors vectors, String path) {
        writeParagraphVectors(vectors, new File(path));
    }

    /**
     * This method saves Word2Vec model into compressed zip file and sends it to output stream
     * PLEASE NOTE: This method saves FULL model, including syn0 AND syn1
     *
     */
    public static void writeWord2VecModel(Word2Vec vectors, File file) {
        try (FileOutputStream fos = new FileOutputStream(file);
             BufferedOutputStream stream = new BufferedOutputStream(fos)) {
            writeWord2VecModel(vectors, stream);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method saves Word2Vec model into compressed zip file and sends it to output stream
     * PLEASE NOTE: This method saves FULL model, including syn0 AND syn1
     *
     */
    public static void writeWord2VecModel(Word2Vec vectors, String path) {
        writeWord2VecModel(vectors, new File(path));
    }

    /**
     * This method saves Word2Vec model into compressed zip file and sends it to output stream
     * PLEASE NOTE: This method saves FULL model, including syn0 AND syn1
     *
     */
    public static void writeWord2VecModel(Word2Vec vectors, OutputStream stream) throws IOException {
        ZipOutputStream zipfile = new ZipOutputStream(new BufferedOutputStream(new CloseShieldOutputStream(stream)));

        ZipEntry syn0 = new ZipEntry("syn0.txt");
        zipfile.putNextEntry(syn0);

        // writing out syn0
        File tempFileSyn0 = DL4JFileUtils.createTempFile("word2vec", "0");
        File tempFileSyn1 = DL4JFileUtils.createTempFile("word2vec", "1");
        File tempFileSyn1Neg = DL4JFileUtils.createTempFile("word2vec", "n");
        File tempFileCodes = DL4JFileUtils.createTempFile("word2vec", "h");
        File tempFileHuffman = DL4JFileUtils.createTempFile("word2vec", "h");
        File tempFileFreqs = DL4JFileUtils.createTempFile("word2vec", "f");
        tempFileSyn0.deleteOnExit();
        tempFileSyn1.deleteOnExit();
        tempFileSyn1Neg.deleteOnExit();
        tempFileFreqs.deleteOnExit();
        tempFileCodes.deleteOnExit();
        tempFileHuffman.deleteOnExit();

        try {
            writeWordVectors(vectors.lookupTable(), tempFileSyn0);

            FileUtils.copyFile(tempFileSyn0, zipfile);

            // writing out syn1
            INDArray syn1 = ((InMemoryLookupTable<VocabWord>) vectors.getLookupTable()).getSyn1();

            if (syn1 != null)
                try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileSyn1))) {
                    for (int x = 0; x < syn1.rows(); x++) {
                        INDArray row = syn1.getRow(x);
                        StringBuilder builder = new StringBuilder();
                        for (int i = 0; i < row.length(); i++) {
                            builder.append(row.getDouble(i)).append(" ");
                        }
                        writer.println(builder.toString().trim());
                    }
                }

            ZipEntry zSyn1 = new ZipEntry("syn1.txt");
            zipfile.putNextEntry(zSyn1);

            FileUtils.copyFile(tempFileSyn1, zipfile);

            // writing out syn1
            INDArray syn1Neg = ((InMemoryLookupTable<VocabWord>) vectors.getLookupTable()).getSyn1Neg();

            if (syn1Neg != null)
                try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileSyn1Neg))) {
                    for (int x = 0; x < syn1Neg.rows(); x++) {
                        INDArray row = syn1Neg.getRow(x);
                        StringBuilder builder = new StringBuilder();
                        for (int i = 0; i < row.length(); i++) {
                            builder.append(row.getDouble(i)).append(" ");
                        }
                        writer.println(builder.toString().trim());
                    }
                }

            ZipEntry zSyn1Neg = new ZipEntry("syn1Neg.txt");
            zipfile.putNextEntry(zSyn1Neg);

            FileUtils.copyFile(tempFileSyn1Neg, zipfile);


            ZipEntry hC = new ZipEntry("codes.txt");
            zipfile.putNextEntry(hC);

            // writing out huffman tree
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileCodes))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    StringBuilder builder = new StringBuilder(encodeB64(word.getLabel())).append(" ");
                    for (int code : word.getCodes()) {
                        builder.append(code).append(" ");
                    }

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileCodes, zipfile);

            ZipEntry hP = new ZipEntry("huffman.txt");
            zipfile.putNextEntry(hP);

            // writing out huffman tree
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileHuffman))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    StringBuilder builder = new StringBuilder(encodeB64(word.getLabel())).append(" ");
                    for (int point : word.getPoints()) {
                        builder.append(point).append(" ");
                    }

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileHuffman, zipfile);

            ZipEntry hF = new ZipEntry("frequencies.txt");
            zipfile.putNextEntry(hF);

            // writing out word frequencies
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileFreqs))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    StringBuilder builder = new StringBuilder(encodeB64(word.getLabel())).append(" ")
                            .append(word.getElementFrequency()).append(" ")
                            .append(vectors.getVocab().docAppearedIn(word.getLabel()));

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileFreqs, zipfile);

            ZipEntry config = new ZipEntry("config.json");
            zipfile.putNextEntry(config);
            //log.info("Current config: {}", vectors.getConfiguration().toJson());
            try (ByteArrayInputStream bais = new ByteArrayInputStream(vectors.getConfiguration().toJson().getBytes(StandardCharsets.UTF_8))) {
                IOUtils.copy(bais, zipfile);
            }

            zipfile.flush();
            zipfile.close();
        } finally {
            for(File f : new File[]{tempFileSyn0, tempFileSyn1, tempFileSyn1Neg, tempFileCodes, tempFileHuffman, tempFileFreqs}){
                try{
                    f.delete();
                } catch (Exception e){
                    //Ignore, is temp file
                }
            }
        }
    }

    /**
     * This method saves ParagraphVectors model into compressed zip file and sends it to output stream
     */
    public static void writeParagraphVectors(ParagraphVectors vectors, OutputStream stream) throws IOException {
        ZipOutputStream zipfile = new ZipOutputStream(new BufferedOutputStream(new CloseShieldOutputStream(stream)));

        ZipEntry syn0 = new ZipEntry("syn0.txt");
        zipfile.putNextEntry(syn0);

        // writing out syn0
        File tempFileSyn0 = DL4JFileUtils.createTempFile("paravec", "0");
        File tempFileSyn1 = DL4JFileUtils.createTempFile("paravec", "1");
        File tempFileCodes = DL4JFileUtils.createTempFile("paravec", "h");
        File tempFileHuffman = DL4JFileUtils.createTempFile("paravec", "h");
        File tempFileFreqs = DL4JFileUtils.createTempFile("paravec", "h");
        tempFileSyn0.deleteOnExit();
        tempFileSyn1.deleteOnExit();
        tempFileCodes.deleteOnExit();
        tempFileHuffman.deleteOnExit();
        tempFileFreqs.deleteOnExit();

        try {

            writeWordVectors(vectors.lookupTable(), tempFileSyn0);

            FileUtils.copyFile(tempFileSyn0, zipfile);

            // writing out syn1
            INDArray syn1 = ((InMemoryLookupTable<VocabWord>) vectors.getLookupTable()).getSyn1();

            if (syn1 != null)
                try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileSyn1))) {
                    for (int x = 0; x < syn1.rows(); x++) {
                        INDArray row = syn1.getRow(x);
                        StringBuilder builder = new StringBuilder();
                        for (int i = 0; i < row.length(); i++) {
                            builder.append(row.getDouble(i)).append(" ");
                        }
                        writer.println(builder.toString().trim());
                    }
                }

            ZipEntry zSyn1 = new ZipEntry("syn1.txt");
            zipfile.putNextEntry(zSyn1);

            FileUtils.copyFile(tempFileSyn1, zipfile);

            ZipEntry hC = new ZipEntry("codes.txt");
            zipfile.putNextEntry(hC);

            // writing out huffman tree
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileCodes))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    StringBuilder builder = new StringBuilder(encodeB64(word.getLabel())).append(" ");
                    for (int code : word.getCodes()) {
                        builder.append(code).append(" ");
                    }

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileCodes, zipfile);

            ZipEntry hP = new ZipEntry("huffman.txt");
            zipfile.putNextEntry(hP);

            // writing out huffman tree
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileHuffman))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    StringBuilder builder = new StringBuilder(encodeB64(word.getLabel())).append(" ");
                    for (int point : word.getPoints()) {
                        builder.append(point).append(" ");
                    }

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileHuffman, zipfile);

            ZipEntry config = new ZipEntry("config.json");
            zipfile.putNextEntry(config);
            IOUtils.write(vectors.getConfiguration().toJson(), zipfile, StandardCharsets.UTF_8);


            ZipEntry labels = new ZipEntry("labels.txt");
            zipfile.putNextEntry(labels);
            StringBuilder builder = new StringBuilder();
            for (VocabWord word : vectors.getVocab().tokens()) {
                if (word.isLabel())
                    builder.append(encodeB64(word.getLabel())).append("\n");
            }
            IOUtils.write(builder.toString().trim(), zipfile, StandardCharsets.UTF_8);

            ZipEntry hF = new ZipEntry("frequencies.txt");
            zipfile.putNextEntry(hF);

            // writing out word frequencies
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileFreqs))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    builder = new StringBuilder(encodeB64(word.getLabel())).append(" ").append(word.getElementFrequency())
                            .append(" ").append(vectors.getVocab().docAppearedIn(word.getLabel()));

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileFreqs, zipfile);

            zipfile.flush();
            zipfile.close();
        } finally {
            for(File f : new File[]{tempFileSyn0, tempFileSyn1, tempFileCodes, tempFileHuffman, tempFileFreqs}){
                try{
                    f.delete();
                } catch (Exception e){
                    //Ignore, is temp file
                }
            }
        }
    }

}
