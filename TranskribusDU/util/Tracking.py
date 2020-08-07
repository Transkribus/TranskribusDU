# -*- coding: utf-8 -*-

"""
Encapsulate tracking with MLFlow, if MLFlow available

Created on 23/12/2019

Copyright NAVER LABS Europe 2019

@author: JL Meunier
"""
import sys, os, tempfile, stat, glob

try:
    import mlflow
except ImportError:
    mlflow = None       # this prevent setting tracking ON

try:
    from common.trace import traceln
except ImportError:
    def traceln(*o): print(*o, file=sys.stderr, flush=True)

# Either load the config from the application PYTHONPATH or from this distro
try:
    import Tracking_config
except ModuleNotFoundError:
    import util.Tracking_config as Tracking_config
    
DEFAULT_URI = "http://%s:%d" % (Tracking_config.sMLFlowHost , Tracking_config.iMLFlowPort)


# MAIN VARIABLE to switch On/OFF the actual tracking to the MLFLOW server
bTracking = False   # tracking off by default

# to hide the underlying exception
try:
    TrackingException = mlflow.exceptions.MlflowException
except AttributeError: 
    TrackingException = Exception

# -------------   TRACKING API   -------------
def set_tracking():
    """
    Enable tracking
    """
    global bTracking
    if mlflow: 
        bTracking = True
    else:
        traceln("ERROR: mlflow not installed")

def set_tracking_uri(server_uri=None):
    """
    Enable the tracking with given MLFlow server URI
    """    
    if mlflow: 
        if server_uri is None: server_uri = DEFAULT_URI
        traceln("MLFLow server: ", server_uri)
        mlflow.set_tracking_uri(server_uri)
        set_tracking()
    
def set_no_tracking():
    """
    Disable tracking
    """
    global bTracking
    bTracking = False

# ----  Setting experiment and start/stop of runs   ----    
def set_experiment(experiment_name):
    if bTracking: mlflow.set_experiment(experiment_name)
            
def start_run(run_name=None):
    # mlflow.start_run(run_id=None, experiment_id=None, run_name=None, nested=False)
    if bTracking and mlflow: 
        for i in range(5): # max retry...
            _s = run_name if i == 0 else "%s.%d" % (run_name, i)
            try:    
                return mlflow.start_run(run_name=_s)
                break
            except: 
                mlflow.end_run()
                traceln("MLFLOW: previous run '%s' probably crashed. Need to generate new name." % _s)
        return None
    else:
        return _NullContextManager()

def end_run(status='FINISHED'):
    if bTracking: mlflow.end_run(status=status)

# ----  Logging parameters, metrics and artifacts   ----    
def log_param(key, value):
    if bTracking: mlflow.log_param(key, value)
    
def log_params(params):
    if bTracking: 
        try:
            mlflow.log_params(params)
        except mlflow.exceptions.MlflowException:
            # for the case of "had length 1296, which exceeded length limit of 250""
            # ... pffff
            for _k,_v in params.items(): log_param(_k,_v)

def log_metric(key, value, step=None
               , ndigits=None):
    """
    Extra parameter: ndigits : if specified, all values are rounded with the given number of digits
    """
    if bTracking:
        if ndigits is None: 
            mlflow.log_metric(key, value, step=step)
        else:
            try: value = round(value, ndigits)
            except: pass
            mlflow.log_metric(key, value, step=step)
            
    
def log_metrics(metrics, step=None
                , ndigits=None):
    """
    Extra parameter: ndigits : if specified, all values are rounded with the given number of digits
    """
    if bTracking: 
        if ndigits is None:
            mlflow.log_metrics(metrics, step=step)
        else:
            _d = {}
            for k,v in metrics.items():
                try: v = round(v, ndigits)
                except: pass
                _d[k] = v
            mlflow.log_metrics(_d, step=step)


def log_artifact(local_path, artifact_path=None):
    if bTracking: 
        _chmod_rw_rw_r(local_path)
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        
def log_artifacts(local_dir, artifact_path=None):
    if bTracking: 
        for fn in glob.iglob(os.path.join(local_dir, "**"), recursive=True):
            _chmod_rw_rw_r(fn)
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

def log_artifact_string(sName, sData):
    """
    make the string a temporary file, log it, delete the file...
    """
    fd, name = tempfile.mkstemp(prefix=(sName+"."), suffix=".txt")
    try:
        os.write(fd, str(sData).encode('utf-8'))
        os.fsync(fd)
        os.close(fd)
        log_artifact(name)
        os.remove(name)
    finally:
        # os.remove(name)
        pass
               
def set_tag(key, value):
    if bTracking: mlflow.set_tag(key, value)
    
def set_tags(tags):
    if bTracking: mlflow.set_tags(tags)


# -----  INTERNAL STUFF --------------------------------------------------
class _NullContextManager(object):
    """
    A context manager that does nothing.
    """
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass

def _chmod_rw_rw_r(fname):
    """
    when used by a group of users:
    - they must be in the same user group
    - the file copied to the server area must be RW by the server, which 
        possibly runs under another account, of the same group of course!
    Here we chose to set RW for user and group, and R for other
    """
    os.chmod(fname, stat.S_IRUSR | stat.S_IWUSR \
                  | stat.S_IRGRP | stat.S_IWGRP \
                  | stat.S_IROTH )
    

# ------------------------------------------------------------------------
def test_no_mlflow():
    global mlflow
    mlflow = None

    start_run("DU.crf.Test.JL")
    log_param("toto", "999")
    log_metric("score", 10, 1)
    log_metric("score", 20, 2)
    end_run()
    print("test_no_mlflow: DONE")

def test_no_mlflow_with():
    global mlflow
    mlflow = None

    with start_run("DU.crf.Test.JL") as rrr:
        log_param("toto", "999")
        log_metric("score", 10, 1)
        log_metric("score", 20, 2)
        end_run()
        print("test_no_mlflow: DONE")
    
def test_simple():
    set_tracking()
    set_experiment("DU.crf.Test.JL")
    start_run("run_1")
    log_param("toto", "999")
    log_metric("score", 10, 1)
    log_metric("score", 20, 2)
    set_tag("k", "vv")
    # log_artifact("dtw.py")
    log_artifact_string("mydata", """Dummy data
in multiline style
""")
    end_run()
    print("test_simple: DONE")

def test_uri():
    import time
    set_tracking_uri("http://cumin.int.europe.naverlabs.com:5000")
    set_experiment("DU.crf.Test.JL")
    start_run("run_%s" % int(time.time()))
    log_param("toto", "999")
    log_metric("score", 10, 1)
    log_metric("score", 20, 2)
    set_tag("k", "vv")
    log_artifact("dtw.py")
    log_artifact_string("mydata", """Dummy data
in multiline style
""")
    end_run()
    print("test_uri: DONE")

def test_api():
    import mlflow, time, os.path
    
    sTestFile = "c:\\tmp\\toto.txt"
    assert os.path.exists(sTestFile)
    
    mlflow.set_tracking_uri("http://cumin.int.europe.naverlabs.com:5000")
    mlflow.set_experiment("test_artifacts")
    mlflow.start_run(run_name="run_%s" % int(time.time()))
    mlflow.log_param("toto", "9.99")
    
    mlflow.log_artifact(sTestFile)
    
    mlflow.end_run()
    
# ------------------------------------------------------------------------
if __name__ == "__main__":
    # test_no_mlflow()
    # test_simple()    
    # test_uri()
    test_api()
