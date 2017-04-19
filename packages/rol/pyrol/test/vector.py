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
        x = vector(self.n)
        return x
    def __str__(self):
        return str(self.data)


if __name__ == '__main__':

    v = vector(10)
    print("type(v) = {0}".format(type(v)))

    v[0]=1.0
    u = v.clone()

    attributes = dir(v)

    print("vector implemented methods:")
    print("dimension   - {0}".format("dimension"   in attributes))
    print("clone       - {0}".format("clone"       in attributes))
    print("__setitem__ - {0}".format("__setitem__" in attributes))
    print("__getitem__ - {0}".format("__getitem__" in attributes))


    print(v)
    print(u)

    
