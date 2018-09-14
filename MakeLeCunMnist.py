import numpy as np
import struct
import matplotlib.pyplot as plt
from numpy.linalg import svd
import glob, os

srcPath = 'D:/ML_Data/LeCun/' #include trailing /
dstPath = 'D:/ML_Data/MNIST_LeCun/' #include trailing /
names = [['train-labels.idx1-ubyte', 'train-images.idx3-ubyte'], ['t10k-labels.idx1-ubyte', 't10k-images.idx3-ubyte']]
batchSize = 256
eps = 1e-6
mtype = np.float32
msize = int(4)
W = np.array([1., 1., 1.], dtype=np.float64) #vector of weights per data dimension - greyscale images images are 3-dimensional (x,y,grey)

#Warning!!! Order of fields in the following class definitions as well as their data types matter since corresponding C++ program relies on it
#Change in these class definitions will require subsequent change in the header SparseVector.h of the C++ code

class KernelParameters:
    def __init__(self, label = 0, rho = 0, gamma = 0, w = W):
        self.rho = np.float64(rho) #when SVM model is saved, thif field is bias of kernel (RBF offset in this case)
        self.gamma = np.float64(gamma) #when SVM model is saved, thif field is parameter of kernel (RBF gamma in this case)
        self.target = np.int32(label) #when SVM model is saved, this field indicates positive class label, otherwise it does not matter
        self.W = w.astype(np.float64)

    def getSize(self):
        return 8+8+4+3*8

    def write(self, f):
        f.write(np.float64(self.rho))
        f.write(np.float64(self.gamma))
        f.write(np.int32(self.target))
        self.W.tofile(f)



class BatchHeader :
    def __init__(self, param = KernelParameters(), numRecords = 0, totalSize = 0):
        self.numRecords = np.int32(numRecords) #number of records in this batch
        self.totalSize = np.int32(totalSize) #total size of the batch in
        self.param = param

    def getSize(self):
        return 4+4+KernelParameters().getSize()

    def write(self, f):
        f.write(np.int32(self.numRecords))
        f.write(np.int32(self.totalSize))
        self.param.write(f)

    def writeSize(self, f, s):
        f.seek(4,0)
        f.write(np.uint32(s))



class ImageHeader :
    def __init__(self, n = 0):
        self.numAfter = np.int32(n) # number of records after this one in the batch
        self.imgSize = np.int32(0) # size of the image record, including header fields
        self.rows = np.int32(0) # number of rows in the data matrix
        self.cols = np.int32(0) # number of columns in the data matrix (3 for images - x,y and greyscale)
        self.alpha = np.float32(0) # alpha = 0
        self.label = np.float32(0) # class label

    def getSize(self):
        return 4+4+4+4+4+4

    def write(self, f):
        f.write(np.int32(self.numAfter))
        f.write(np.int32(self.imgSize))
        f.write(np.int32(self.rows))
        f.write(np.int32(self.cols))
        f.write(np.float32(self.alpha))
        f.write(np.float32(self.label))



class ImageData :
    def __init__(self, d = np.array([]), h = ImageHeader(), y = 0):
        self.header = h
        self.header.rows = d.shape[0] #these rows and columns are different than the original 28x28 since we convert images to scatter plots (as they really are)
        self.header.cols = d.shape[1]
        self.header.label = y
        self.data = d.reshape(1,np.prod(d.size)) #linearize matrix

        #padding is required in order provide alignment of the data in CUDA device memory
        if (np.mod(np.prod(d.shape), msize) != 0):
            pad = msize - np.mod(np.prod(d.shape), msize)
            self.data = np.hstack((self.data, np.zeros((1,pad)))).astype(mtype)

        self.header.imgSize = np.int32(ImageHeader().getSize()+self.data.shape[1]*msize)

    def write(self, f):
        self.header.write(f)
        self.data.tofile(f)

    def getSize(self):
        return self.header.imgSize


def readNextImage(fx, fy, rows, cols) :
    l = fy.read(1)
    l = int(struct.unpack('B', l)[0])
    x = fx.read(rows*cols)
    r = np.array([float(b) for b in x]).reshape((rows, cols))
    pos = np.where(r)
    pos = np.vstack((pos[0],pos[1])).T
    x = r[pos[:,0],pos[:,1]]
    x = x[:,None]
    d = np.hstack((pos, x))
    d = (d - np.mean(d, axis=0))/(eps + np.std(d, axis=0))
    d = d.astype(mtype)
    return d,l


def writeData(nameL, nameX, prefix, descew = False) :

    fy = open(srcPath + nameL, 'rb')
    fx = open(srcPath + nameX, 'rb')


    magic = fy.read(4)
    numitems = fy.read(4)

    magic = fx.read(4)
    numitems = fx.read(4)
    rows = fx.read(4)
    cols = fx.read(4)
    rows = int(struct.unpack('i', rows[::-1])[0])
    cols = int(struct.unpack('i', cols[::-1])[0])

    magic = int(struct.unpack('i', magic[::-1])[0])
    numitems = int(struct.unpack('i', numitems[::-1])[0])

    N = int((numitems + batchSize - 1) / batchSize)
    last = np.mod(numitems, batchSize)
    S = [batchSize]*(N-1) + [last]

    fig,ax = plt.subplots(1)
    fig.show()

    for i,n in zip(range(N), S):
        print(i)
        size = BatchHeader().getSize()
        with open(dstPath + prefix + '_{}.dat'.format(i), 'wb') as f:
            BatchHeader(KernelParameters(), n).write(f)

            for a in np.arange(n, 0, -1) :
                d,l = readNextImage(fx,fy,rows,cols)

                #descew each image individually using SVD
                if descew:
                    u,s,vh = svd(d[:,:2], full_matrices=False)
                    u = (u - np.mean(u, axis=0))/(eps + np.std(u, axis=0))
                    u = np.dot(u, vh)
                    d = np.hstack((u,d[:,2][:,None])).astype(mtype)

                if l == 30: #change to visualize appropriate digit
                    ax.clear()
                    ax.scatter(d[:,1], -d[:,0], c=d[:,2], cmap=plt.cm.binary, marker='s')
                    plt.xlim((-2,2))
                    plt.ylim((-2,2))
                    plt.title(str(l))

                data = ImageData(d, ImageHeader(a-1), l)
                data.write(f)
                size += data.getSize()

        with open(dstPath + prefix + '_{}.dat'.format(i), 'r+b') as f:
            BatchHeader().writeSize(f, size)

        if i == 2:
            return



for f in glob.glob(dstPath + '*.dat'):
    os.remove(f)


writeData(names[0][0], names[0][1], 'TRAIN', True)
writeData(names[1][0], names[1][1], 'TEST', True)