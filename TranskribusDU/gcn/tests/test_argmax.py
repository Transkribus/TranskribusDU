import numpy as np

# -----  TESTS ---------------------
def test_argmax():
    a = np.array([  [0, 1, 2]
                  , [2, 0, 0]
                  , [0, 1, 0]])
    v = a.argmax(axis=1)
    assert v.tolist() == [2, 0, 1], v
    assert (v == np.array([2, 0, 1])).all()
    
if __name__ == "__main__":
    test_argmax()
    