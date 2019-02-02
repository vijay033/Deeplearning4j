package com.example.vijay.ondevice_word2vector;

import android.content.Context;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.IOException;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private String TAG = "Word2Vec";
    TextView tvWord, tvSimilarwords;
    EditText edText;
    Button edtButton;
    String edString;
    ProgressBar prgbar;
    WordVectorTraining wordVectorTraining;
    Context context;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        edText = (EditText) findViewById(R.id.edWord);
        edtButton = (Button)findViewById(R.id.edButton);
        tvWord = (TextView)findViewById(R.id.edString);
        tvSimilarwords = (TextView)findViewById(R.id.edSimilar);
        prgbar = (ProgressBar)findViewById(R.id.progressBar);
        prgbar.setVisibility(View.INVISIBLE);
        context = this.getApplicationContext();

        try{
            wordVectorTraining = new WordVectorTraining(context);
        }catch (IOException e){
            e.printStackTrace();
        }

        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                prgbar.setVisibility(View.VISIBLE);
                wordVectorTraining.trainW2V();
                prgbar.post(new Runnable() {
                    @Override
                    public void run() {
                        Word2Vec []vec = wordVectorTraining.getW2VInstance();
                        if(vec.length > 0){
                            tvWord.setText("SUCCESS");
                        }else{
                            tvWord.setText("FAILED");
                        }
                        prgbar.setVisibility(View.INVISIBLE);
                    }
                });
            }
        };
        new Thread(runnable).start();

        edtButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                edString = edText.getText().toString();

                Word2Vec []vec = wordVectorTraining.getW2VInstance();

                if(vec.length > 0){
                    tvWord.setText(edString);
                    String tvdisplay = "";
                    for(int i = 0 ; i < vec.length ; i++) {
                        double[] vectors = vec[i].getWordVector(edString);
                        if(vectors != null && vectors.length > 0){
                            ArrayList<String> lst = (ArrayList<String>)vec[i].wordsNearest(edString,5);
                            for(int j = 0; j < lst.size(); j++){
                                tvdisplay += '#';
                                tvdisplay += lst.get(i);
                            }
                            tvSimilarwords.setText(tvdisplay);
                        }
                    }
                    if(tvdisplay.equals("")){
                        tvSimilarwords.setText("No Vector Found In Database");
                    }
                }
            }
        });

    }
}
