import numpy as np
from sklearn import datasets as ds

class MOONS:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):

        trn, val, tst = load_data()

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

        
def load_data():
    x = ds.make_moons(n_samples=30000, shuffle=True, noise=0.05)[0]
    return x[:24000], x[24000:27000], x[27000:]
