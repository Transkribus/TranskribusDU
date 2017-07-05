Experiments with some tabular text from the archive of the abbey of Passau (EU READ partner)

H. Dejean, JL. Meunier
May-June 2017

----------------------------------------------------------------------------------------------------
TASK
The global task consists in segmenting each page in table cells.
The segmentation in columns is dealt with by a template-based method, which is not discussed here.
The segmenting in table rows is done using CRF (supervised ML).

----------------------------------------------------------------------------------------------------
DATA
H. Dejean prepared the data and annotated it. The corresponding PageXml XML files are in abp/col folder.
We have 144 documents, each of 1 page containing 1 table. 
Each page contains text lines as well as some graphical lines obtained from an automatic line detector provided by a READ partner.
The annotation:
- for text lines: we use the conventional BIES (Begin, Inside, End, Singleton) labeling scheme (often used in NLP). The first text line of a cell is B, the last is E, and any text line in-between is I. A text line occupying alone a cell is S.
- for graphical lines, we use a binary annotation to indicate if the line is separating two sets of cells or if it is noise (with respect to the task).

The annotation is in the XML @type attributes.

----------------------------------------------------------------------------------------------------
METHOD
We use here the graph CRF and multi-type graph CRF supervised methods. 

For CRF, we only consider text lines to create the CRF grap nodes. Edges denote an horizontal or vertical neighborohood between two text lines.
For multi-type CRF, we additionally consider the graphical lines as CRF nodes. We therefore have 2 types of nodes in our graph, and consequently 4 types of edges. In addition to the edges defined above between text lines, we create the edges due to the same neighborhood relation ship between text lines and graphical lines, and vice-versa, and among graphical lines. The single-type CRF graph is a sub-graph of the multi-type CRF graph.

Features for nodes and edges are essentially geometric ones. We do not use the textual content of text lines. In particular, we produce the same sort of features for graphical lines and text lines.

----------------------------------------------------------------------------------------------------
EVALUATION
We run a 10-folds cross-validation. Each fold contains several documents. We then compute the standard precision and recall per node label per fold. We then compare the F1 measure of each method. 

----------------------------------------------------------------------------------------------------
SCRIPT to reproduce the results (RR)

	#creation of the same 10 folds for the single- and multi-type experiments.
	python /opt/project/read/jl_git/TranskribusDU/src/tasks/DU_ABPTable.py --fold-init 10 abp_models abp_CV10 --fold abp 
	#copying their definition
	mkdir tmp_def_fold
	cp abp_models/abp_CV10_fold*def* tmp_def_fold
	rename abp_CV10 abp_T_CV10 tmp_def_fold/*
	mv -i tmp_def_fold/* abp_models/
	rmdir tmp_def_fold

	#SINGLE CV10 ABP
	for i in 1 2 3 4 5 6 7 8 9 10
	do
		python /opt/project/read/jl_git/TranskribusDU/src/tasks/DU_ABPTable.py --fold-run $i  abp_models abp_CV10 --crf-njobs=20 > abp_CV10_log_fold_$i.log 2>&1 &
	done
	#aggregation of the results
	python /opt/project/read/jl_git/TranskribusDU/src/tasks/DU_ABPTable.py --fold-finish  abp_models abp_CV10 
	#result is in abp_models/abp_CV10_folds_STATS.txt
	
	#MULTI CV10 ABP
	for i in 1 2 3 4 5 6 7 8 9 10
	do
		python /opt/project/read/jl_git/TranskribusDU/src/tasks/DU_ABPTable_T.py --fold-run $i abp_models abp_T_CV10 --crf-njobs=20 > abp_T_CV10_log_fold_$i.log 2>&1 &
	done
	python /opt/project/read/jl_git/TranskribusDU/src/tasks/DU_ABPTable_T.py --fold-finish abp_models abp_T_CV10 
	#result is in abp_models/abp_T_CV10_folds_STATS.txt
	
	#convenience to sump-up the results
	cd abp_models
	for lbl in RB RI RE RS; do egrep "^     abp_$lbl" abp_CV10_folds_STATS.txt   |head -10 >> abp_CV10_folds_STATS_by_lbl.txt; done
	for lbl in RB RI RE RS; do egrep "^    text_$lbl" abp_T_CV10_folds_STATS.txt |head -10 >> abp_T_CV10_folds_STATS_by_lbl.txt; done
	

----------------------------------------------------------------------------------------------------
RESULTS

--- Precision, recall, F1, per label
- SINGLE-TYPE
             precision    recall  f1-score   support

     abp_RB      0.894     0.914     0.904     24310
     abp_RI      0.808     0.902     0.853     22806
     abp_RE      0.888     0.908     0.898     24296
     abp_RS      0.887     0.745     0.810     22199
     abp_RO      0.058     0.033     0.042       459

avg / total      0.866     0.865     0.864     94070

- MULTI-TYPE
             precision    recall  f1-score   support

    text_RB      0.916     0.925     0.921     24310
    text_RI      0.825     0.918     0.869     22806
    text_RE      0.910     0.915     0.912     24296
    text_RS      0.892     0.775     0.829     22199
    text_RO      0.036     0.035     0.036       459
   sprtr_SI      0.777     0.805     0.791     14194
   sprtr_SO      0.895     0.879     0.887     26973

avg / total      0.874     0.872     0.872    135237

The difference is statistically significant (paired difference test using Wilcoxon's test).

--- confusion matrices
- SINGLE-TYPE
  Line=True class, column=Prediction
abp_RB  [[22215  1064   163   806    62]
abp_RI   [  885 20582   969   347    23]
abp_RE   [  128  1134 22065   894    75]
abp_RS   [ 1465  2656  1458 16538    82]
abp_RO   [  165    38   189    52    15]]
(unweighted) Accuracy score = 0.87     trace=81415  sum=94070

- MULTI-TYPE
  Line=True class, column=Prediction
 text_RB  [[22495   800    99   831    85     0     0]
 text_RI   [  707 20945   725   357    72     0     0]
 text_RE   [   76  1057 22221   835   107     0     0]
 text_RS   [ 1114  2533  1196 17194   162     0     0]
 text_RO   [  161    56   172    54    16     0     0]
sprtr_SI   [    0     0     0     0     0 11420  2774]
sprtr_SO   [    0     0     0     0     0  3277 23696]]
(unweighted) Accuracy score = 0.87     trace=117987  sum=135237
