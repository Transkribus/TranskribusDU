#!/usr/bin/env bash

python DU_gcn_task.py --das_predict=True --configid=31 --das_predict_workflow=True --das_predict_earlystop=True
python DU_gcn_task.py --das_predict=True --configid=5 --das_predict_workflow=True --das_predict_earlystop=True
python DU_gcn_task.py --das_predict=True --configid=33 --das_predict_workflow=True --das_predict_earlystop=True


python DU_gcn_task.py --das_predict=True --configid=31 --das_predict_earlystop=True
python DU_gcn_task.py --das_predict=True --configid=5  --das_predict_earlystop=True
python DU_gcn_task.py --das_predict=True --configid=33 --das_predict_earlystop=True
python DU_gcn_task.py --das_predict=True --configid=42 --das_predict_earlystop=True
python DU_gcn_task.py --das_predict=True --configid=42 --das_predict_workflow=True --das_predict_earlystop=True



python DU_gcn_task.py --das_predict=True --configid=47 --das_predict_workflow=True
python DU_gcn_task.py --das_predict=True --configid=47