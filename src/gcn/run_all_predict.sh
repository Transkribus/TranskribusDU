#!/usr/bin/env bash

python DU_gcn_task.py --das_predict=True --configid=31 --das_predict_workflow=True
python DU_gcn_task.py --das_predict=True --configid=5 --das_predict_workflow=True
python DU_gcn_task.py --das_predict=True --configid=33 --das_predict_workflow=True


python DU_gcn_task.py --das_predict=True --configid=31
python DU_gcn_task.py --das_predict=True --configid=5
python DU_gcn_task.py --das_predict=True --configid=33
