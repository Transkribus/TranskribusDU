

PY_PATH="/local/TranskribusDU/TranskribusDU/"
TKB_PATH="/local/TranskribusDU/"

ecn_8L1C:
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn8Lay1C --fold-run 1  --ecn --ecn_config ecn_8Lay1Conv.json
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn8Lay1C --fold-run 2  --ecn --ecn_config ecn_8Lay1Conv.json
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn8Lay1C --fold-run 3  --ecn --ecn_config ecn_8Lay1Conv.json
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn8Lay1C --fold-run 4  --ecn --ecn_config ecn_8Lay1Conv.json

ecn_ensemble:
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn_ensemble --fold-run 1  --ecn --ecn_config ecn_ensemble_test.json 
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn_ensemble --fold-run 2  --ecn --ecn_config  ecn_ensemble_test.json 
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn_ensemble --fold-run 3  --ecn --ecn_config  ecn_ensemble_test.json 
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn_ensemble --fold-run 4  --ecn --ecn_config  ecn_ensemble_test.json


ecn_ensemble_stack:
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn_ensemble_stack --fold-run 1  --ecn --ecn_config  ecn_ensemble_3L5C.json	
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn_ensemble_stack --fold-run 2  --ecn --ecn_config  ecn_ensemble_3L5C.json  
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn_ensemble_stack --fold-run 3  --ecn --ecn_config  ecn_ensemble_3L5C.json 
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn_ensemble_stack --fold-run 4  --ecn --ecn_config  ecn_ensemble_3L5C.json

ecn_8L1C_all:
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn8Lay1C_allfold --trn abp --tst abp_DAS_col9142/  --ecn --ecn_config ecn_8Lay1Conv.json
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn8Lay1C_allfold --tst abp_DAS_col9142 --tst abp_DAS_col9142/  --ecn --ecn_config ecn_8Lay1Conv.json

ecn_ensemble_all:
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn_ensemble_allfold --trn abp   --ecn --ecn_config ecn_ensemble_test.json 
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn_ensemble_allfold --tst abp_DAS_col9142 --tst abp_DAS_col9142/  --ecn --ecn_config ecn_ensemble_test.json


ecn_8L1C_SA:
        PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn8Lay1C_SA --fold-run 1  --ecn --ecn_config ecn_8Lay1Conv_sa.json
        PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn8Lay1C_SA --fold-run 2  --ecn --ecn_config ecn_8Lay1Conv_sa.json
        PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn8Lay1C_SA --fold-run 3  --ecn --ecn_config ecn_8Lay1Conv_sa.json
        PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn8Lay1C_SA --fold-run 4  --ecn --ecn_config ecn_8Lay1Conv_sa.json

#rename  rename .xml .mpxml *.xml

ecn_8L1C_all_SA:
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn8Lay1C_allfold_SA --trn abp --tst abp_DAS_col9142/  --ecn --ecn_config ecn_8Lay1Conv_sa.json
	PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn8Lay1C_allfold_SA --tst abp_DAS_col9142 --tst abp_DAS_col9142/  --ecn --ecn_config ecn_8Lay1Conv_sa.json




ecn_logit_attention:
        PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn_logit_attn --fold-run 1  --ecn --ecn_config ecn_logit_attention.json
        PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn_logit_attn --fold-run 2  --ecn --ecn_config ecn_logit_attention.json
        PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn_logit_attn --fold-run 3  --ecn --ecn_config ecn_logit_attention.json
        PYTHONPATH="/local/TranskribusDU/TranskribusDU/" python /local/TranskribusDU/TranskribusDU/tasks/DU_ABPTable.py ABP_EXP  ecn_logit_attn --fold-run 4  --ecn --ecn_config ecn_logit_attention.json



