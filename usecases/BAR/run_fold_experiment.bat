set ExpName=%1
set LabelType=%2

#split folders in N folds
python c:\Local\meunier\git\TranskribusDU\src\tasks\DU_BAR_%LabelType%.py --fold-init=3 cross_valid_models_%LabelType% %ExpName% --fold GT100

#clean any previous model   DANGEROUS!!!!
del cross_valid_models_%LabelType%\*model*

#run each fold
python c:\Local\meunier\git\TranskribusDU\src\tasks\DU_BAR_%LabelType%.py --fold-run=1  cross_valid_models_%LabelType% %ExpName% --crf-max_iter=125 --crf-njobs=3 --crf-C=0.01
python c:\Local\meunier\git\TranskribusDU\src\tasks\DU_BAR_%LabelType%.py --fold-run=2  cross_valid_models_%LabelType% %ExpName% --crf-max_iter=125 --crf-njobs=3 --crf-C=0.01
python c:\Local\meunier\git\TranskribusDU\src\tasks\DU_BAR_%LabelType%.py --fold-run=3  cross_valid_models_%LabelType% %ExpName% --crf-max_iter=125 --crf-njobs=3 --crf-C=0.01

#show model loss over iterations for each run
python c:\Local\meunier\git\TranskribusDU\src\crf\Model_SSVM_AD3.py cross_valid_models_%LabelType% %ExpName%_fold_1
python c:\Local\meunier\git\TranskribusDU\src\crf\Model_SSVM_AD3.py cross_valid_models_%LabelType% %ExpName%_fold_2
python c:\Local\meunier\git\TranskribusDU\src\crf\Model_SSVM_AD3.py cross_valid_models_%LabelType% %ExpName%_fold_3

#gather results
python c:\Local\meunier\git\TranskribusDU\src\tasks\DU_BAR_%LabelType%.py --fold-finish cross_valid_models_%LabelType% %ExpName%

