#!/usr/bin/env bash
source /opt/project/read/VIRTUALENV_PYTHON_type/bin/activate
cd /opt/MLS_db/usr/sclincha/Transkribus/src/tasks && python Dodge_Tasks.py make_test dodge_test_plan.pickle $1