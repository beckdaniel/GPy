import numpy as np
import pylab as pb
import shelve
import GPy
import sys

#My functions
sys.path.append('../../../playground/malaria')
import useful

pb.ion()
pb.close('all')

#all_stations
malaria_data = shelve.open('../../../playground/malaria/uganda_data_20130213.dat',writeback=False)
all_stations = malaria_data['stations']
malaria_data.close()

#Load data
malaria_data = shelve.open('../../../playground/malaria/uganda_ndvi_20130213.dat',writeback=False)
all_districts = malaria_data['districts']
all_variables = malaria_data['headers']
malaria_data.close()

all_stations = ['Gulu']#,'Arua']#,'Mbarara','Kampala','Kasese']

#Multioutput models outputs/inputs
X1list_ = []
X1list = []
X1list_fut = []

X2list_ = []
X2list = []
X2list_fut = []

Hlist_ = []
Hlist = []
Hlist_fut = []

Y1list = []
Y1list_ = []
Y1list_fut = []

Y2list_ = []
Y2list = []
Y2list_fut = []

YYlist = []
YYlist_ = []
YYlist_fut = []

likelihoods1 = []
likelihoods2 = []
R = 2*len(all_stations)
stations = all_stations
I = np.arange(len(stations))

cut_date = 1400
for i,district in zip(I,stations):
    Y1_,X1_ = useful.filtered(district,'incidence',rm_zero=False)
    Y2_,X2_ = useful.raw(district,'ndvi')

    #cut
    index1=np.arange(X1_.size)
    index2=np.arange(X2_.size)
    cut1 = index1[X1_.flatten() <= cut_date].size
    cut2 = index2[X2_.flatten() <= cut_date].size

    Y1 = Y1_[:cut1,:]
    Y1_fut = Y1_[cut1:,:]
    X1 = X1_[:cut1,:]
    X1_fut = X1_[cut1:,:]

    Y2 = Y2_[:cut2,:]
    Y2_fut = Y2_[cut2:,:]
    X2 = X2_[:cut2,:]
    X2_fut = X2_[cut2:,:]

    Y1list_.append(Y1_)
    Y1list.append(Y1)
    Y1list_fut.append(Y1_fut)
    Y2list_.append(Y2_)
    Y2list.append(Y2)
    Y2list_fut.append(Y2_fut)

    likelihoods1.append(GPy.likelihoods.Gaussian(Y1,normalize=False))
    likelihoods2.append(GPy.likelihoods.Gaussian(Y2,normalize=False))

    #Index time
    X1list_.append(np.hstack([np.repeat(i,X1_.size)[:,None],X1_]))
    X1list.append(np.hstack([np.repeat(i,X1.size)[:,None],X1]))
    X1list_fut.append(np.hstack([np.repeat(i,X1_fut.size)[:,None],X1_fut]))

    X2list_.append(np.hstack([np.repeat(i+len(all_stations),X2_.size)[:,None],X2_]))
    X2list.append(np.hstack([np.repeat(i+len(all_stations),X2.size)[:,None],X2]))
    X2list_fut.append(np.hstack([np.repeat(i+len(all_stations),X2_fut.size)[:,None],X2_fut]))

YYlist_ = Y1list_ + Y2list_
YYlist = Y1list + Y2list
YYlist_fut = Y1list_fut + Y2list_fut

Xlist_ = X1list_ + X2list_
Xlist = X1list + X2list
Xlist_fut = X1list_fut + X2list_fut

likelihoodsH = likelihoods1 + likelihoods2


districts = ['Gulu']#,'Arua']
additional_outputs_d = ['ndvi'] #Don't include weather-stations data here
stations = []
outputs_s = [] #NOTE this example only supports one output_s

outputs_d = ['incidence'] + additional_outputs_d
cut_date = 1400


"""
Sparse multioutput model
"""
print '\nMultioutput model'

R = len(stations)*len(outputs_s) + len(districts)*len(outputs_d)
I = np.arange(len(stations))
Ylist_train = []
Xlist_train = []
Ylist_test = []
Xlist_test = []
likelihoods = []

k = 0 #output index
#Data from districts
for output in outputs_d:
    for district in districts:
        #Geta data
        if output == 'incidence':
            y,x = useful.filtered(district,output,rm_zero=False)
        else:
            y,x = useful.raw(district,output)
        #Train datasets
        xtrain = x[x<=cut_date][:,None]
        Xlist_train.append( np.hstack([np.repeat(k,xtrain.size)[:,None],xtrain]) )
        Ylist_train.append(y[x<=cut_date][:,None])
        likelihoods.append(GPy.likelihoods.Gaussian(Ylist_train[-1],normalize=False))
        #Test datasets
        xtest = x[x>cut_date][:,None]
        Xlist_test.append( np.hstack([np.repeat(k,xtest.size)[:,None],xtest]) )
        Ylist_test.append(y[x>cut_date][:,None])

print 'lengths'
for x1,x2,l1,l2 in zip(Xlist,Xlist_train,likelihoodsH,likelihoods):
    print x1.shape, x2.shape
    pb.figure()
    pb.plot(x1[:,1],x2[:,1],'kx')
    pb.figure()
    pb.plot(l1.Y,l2.Y,'kx')



