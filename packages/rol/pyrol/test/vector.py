from array import array

class vector(object):

    def __init__(self,n):
        self.n=n
        self.data = array('d',[0]*self.n)
    def __setitem__(self,i,value):
        self.data[i] = value
    def __getitem__(self,i):
        return self.data[i]
    def dimension(self):
        return self.n
    def clone(self):
        return array('d',[0]*self.n)
    def __str__(self):
        return str(self.data)



