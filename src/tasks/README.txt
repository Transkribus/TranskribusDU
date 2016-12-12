
SEE https://read02.uibk.ac.at/wiki/index.php/Document_Understanding/First_CRF_Experiment

#Convenience variable, pointing to my source directories
srcDU=/c/Local/meunier/git/TranskribusDU/src
srcTR=/c/Local/meunier/git/TranskribusPyClient/src

#Also, since I'm using cygwin on Windows, I've a python,sh script dealing zith file path conversions... 

#making a TRAINING sandbox collection on Transkribus
#technically, the train and test collections are only read, but not modified. 
#So I could work from the original collection, actually. But...
  > ./python.sh $srcTR/TranskribusCommands/Transkribus_do_createCollec.py READDU_JL_TRN
  --> 3820
#Adding some annotated document to it
  > ./python.sh $srcTR/TranskribusCommands/do_addDocToCollec.py 3571 3820 7749 7750
#Downloading the XML on my machine
  > ./python.sh $srcTR/TranskribusCommands/Transkribus_downloader.py 3820 --noimage
  - Done, see in .\trnskrbs_3820

#got this on disk
  > ls trnskrbs_3820
  col/  config.txt*  out/  ref/  run/  xml/
  > ls trnskrbs_3820/col
  7749/  7749.mpxml*  7749_max.ts*  7750/  7750.mpxml*  7750_max.ts*  trp.json*

#Training!!!
  > ./python.sh $srcDU/tasks/DU_StAZH_a.py --trn trnskrbs_3820 mdl-StAZH_a trn3820

#again with a test collection - also annotated to compute some performance score of the model
  > ./python.sh $srcTR/TranskribusCommands/Transkribus_do_createCollec.py READDU_JL_TST
-->3832
  > ./python.sh $srcTR/TranskribusCommands/do_addDocToCollec.py 3571 3820 8251
  > ./python.sh $srcTR/TranskribusCommands/Transkribus_downloader.py 3832
- Done, see in .\trnskrbs_3832

#TESTING!!!!!!!!!!!!!!!
  > ./python.sh $srcDU/tasks/DU_StAZH_a.py mdl-StAZH_a trn3820 --tst trnskrbs_3832
--------------------------------------------------
Trained model 'mdl-StAZH_a' in folder 'trn3820'
Test  collection(s):['C:\\tmp_READ\\tuto\\trnskrbs_3832\\col']
--------------------------------------------------
- loading a crf.Model_SSVM_AD3.Model_SSVM_AD3 model
        - loading pre-computed data from: trn3820\mdl-StAZH_a_model.pkl
                 file found on disk: trn3820\mdl-StAZH_a_model.pkl
                 file is fresh
        - loading pre-computed data from: trn3820\mdl-StAZH_a_transf.pkl
                 file found on disk: trn3820\mdl-StAZH_a_transf.pkl
                 file is fresh
 done
- classes: ['OTHER', 'catch-word', 'header', 'heading', 'marginalia', 'page-number']
- loading test graphs
        C:\tmp_READ\tuto\trnskrbs_3832\col\8251.mpxml
        - 58 nodes,  75 edges)
 1 graphs loaded
        - computing features on test set
          #features nodes=521  edges=532
         done
        - predicting on test set
         done
Line=True class, column=Prediction
               OTHER  [[21  1  2  0  1]
              header   [ 0  6  2  0  0]
             heading   [ 0  0  1  0  0]
          marginalia   [ 0  0  0 15  0]
         page-number   [ 0  0  0  0  9]]
             precision    recall  f1-score   support

      OTHER       1.00      0.84      0.91        25
     header       0.86      0.75      0.80         8
    heading       0.20      1.00      0.33         1
 marginalia       1.00      1.00      1.00        15
page-number       0.90      1.00      0.95         9

avg / total       0.95      0.90      0.92        58

(unweighted) Accuracy score = 0.90


#APPLYING the model!!!!!!!!!!!!!!!
#Now the collection where I'll apply the model
  > ./python.sh $srcTR/TranskribusCommands/Transkribus_do_createCollec.py READDU_JL_PRD
-->3829
#so here, I copy the documents beause the model will produce a new transcript. 
#(at this stage, I do not want to impact the "real" document.)
$   > ./python.sh $srcTR/TranskribusCommands/do_copyDocToCollec.py 3571 3829 8251 8252 8564-8566
  > ./python.sh $srcTR/TranskribusCommands/Transkribus_downloader.py 3829 --noimage
---> - Done, see in .\trnskrbs_3829

  > ./python.sh $srcDU/tasks/DU_StAZH_a.py mdl-StAZH_a trn3820 --run trnskrbs_3829
--> - done

#we produced some ..._du.mpxml files
$ ls trnskrbs_3829/col
8620/           8620_max.ts*  8621_du.mpxml*  8622.mpxml*     8623/           8623_max.ts*  8624_du.mpxml*
8620.mpxml*     8621/         8621_max.ts*    8622_du.mpxml*  8623.mpxml*     8624/         8624_max.ts*
8620_du.mpxml*  8621.mpxml*   8622/           8622_max.ts*    8623_du.mpxml*  8624.mpxml*   trp.json*

#now upload to Transkribus
  > ./python.sh $srcTR/TranskribusCommands/Transkribus_transcriptUploader.py ./trnskrbs_3829 3829
- DONE, all transcripts were uploaded. See in collection 3829


#Actually, this collection was also annotated, so we can compute a score on it
  > ./python.sh $srcDU/tasks/DU_StAZH_a.py mdl-StAZH_a trn3820 --tst trnskrbs_3829

Line=True class, column=Prediction
               OTHER  [[176  11   6   8   1  13]
          catch-word   [  0   0   0   0   0   0]
              header   [  0   0  38   3   0   2]
             heading   [  0   0   0   2   0   0]
          marginalia   [  0   0   0   0  62   0]
         page-number   [  0   0   0   0   0  48]]
C:\Anaconda2\lib\site-packages\sklearn\metrics\classification.py:1076: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
  'recall', 'true', average, warn_for)
             precision    recall  f1-score   support

      OTHER       1.00      0.82      0.90       215
 catch-word       0.00      0.00      0.00         0
     header       0.86      0.88      0.87        43
    heading       0.15      1.00      0.27         2
 marginalia       0.98      1.00      0.99        62
page-number       0.76      1.00      0.86        48

avg / total       0.95      0.88      0.90       370

(unweighted) Accuracy score = 0.88





  srcDU=/c/Local/meunier/git/TranskribusDU/src
  srcTR=/c/Local/meunier/git/TranskribusPyClient/src

  ./python.sh $srcTR/TranskribusCommands/Transkribus_do_createCollec.py READDU_JL_TRN
  ./python.sh $srcTR/TranskribusCommands/do_addDocToCollec.py 3571 3820 7749 7750
  ./python.sh $srcTR/TranskribusCommands/Transkribus_downloader.py 3820 --noimage
  ./python.sh $srcDU/tasks/DU_StAZH_a.py ./mdl-StAZH_a MyModel --trn trnskrbs_3820


  ./python.sh $srcTR/TranskribusCommands/Transkribus_do_createCollec.py READDU_JL_TST
  ./python.sh $srcTR/TranskribusCommands/do_addDocToCollec.py 3571 3820 8251
  ./python.sh $srcTR/TranskribusCommands/Transkribus_downloader.py 3832
  ./python.sh $srcDU/tasks/DU_StAZH_a.py ./mdl-StAZH_a MyModel --tst trnskrbs_3832

  ./python.sh $srcTR/TranskribusCommands/Transkribus_do_createCollec.py READDU_JL_PRD
  ./python.sh $srcTR/TranskribusCommands/do_copyDocToCollec.py 3571 3829 8251 8252 8564-8566
  ./python.sh $srcTR/TranskribusCommands/Transkribus_downloader.py 3829 --noimage
  ./python.sh $srcDU/tasks/DU_StAZH_a.py ./mdl-StAZH_a MyModel --run trnskrbs_3829
  ./python.sh $srcTR/TranskribusCommands/Transkribus_transcriptUploader.py ./trnskrbs_3829 3829
