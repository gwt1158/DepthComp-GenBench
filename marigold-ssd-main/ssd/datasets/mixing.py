# dataset mixing is adapted from
# https://github.com/VisualComputingInstitute/diffusion-e2e-ft/blob/main/training/dataloaders/load.py
import numpy as np

class MixedDataLoader:
    def __init__(self, loader1, loader2, split1=9, split2=1):
        self.loader1 = loader1
        self.loader2 = loader2
        self.split1 = split1
        self.split2 = split2
        self.frac1, self.frac2 = self.get_split_fractions()
        self.randchoice1=None

    def __iter__(self):
        self.loader_iter1 = iter(self.loader1)
        self.loader_iter2 = iter(self.loader2)
        self.randchoice1 = self.create_split()
        self.indx = 0
        return self
    
    def get_split_fractions(self):
        if self.split1 == 0:
            return 0, 1

        if self.split2 == 0:
            return 1, 0

        size1 = len(self.loader1)
        size2 = len(self.loader2)
        effective_fraction1 = min((size2/size1) * (self.split1/self.split2), 1) 
        effective_fraction2 = min((size1/size2) * (self.split2/self.split1), 1) 
        return effective_fraction1, effective_fraction2

    def create_split(self):
        randchoice1 = [True]*int(len(self.loader1)*self.frac1) + [False]*int(len(self.loader2)*self.frac2)
        np.random.shuffle(randchoice1)
        return randchoice1

    def __next__(self):
        if self.indx == len(self.randchoice1):
            raise StopIteration
        if self.randchoice1[self.indx]:
            self.indx += 1
            return next(self.loader_iter1)
        else:
            self.indx += 1
            return next(self.loader_iter2)
        
    def __len__(self):
        return int(len(self.loader1)*self.frac1) + int(len(self.loader2)*self.frac2)
