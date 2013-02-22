"""
Multioutput GP for malaria counts
---------------------------------
dataset: ../../../playground/malaria/uganda_ndvi_20130213.dat
B matrix controls the relation between districts
Incidences are assumed to have a log-normal distribution
"""
#NOTE This is a non-sparse model

import numpy as np
import pylab as pb
import shelve
import GPy
import datetime

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

def ndvi_data(district_name,variable_name):
    """
    Returns a variable in specific district form file uganda_ndvi_20130213.dat (weekly data)
    """
    #Open data file
    malaria_data = shelve.open('../../../playground/malaria/uganda_ndvi_20130213.dat',writeback=False)
    all_districts = malaria_data['districts']
    all_variables = malaria_data['headers']
    #Get district number in data arrange
    district = [district_name]
    d_number = int(np.arange(len(all_districts))[np.array(all_districts) == district])
    #Get variable number in data arrange
    variable = [variable_name]
    v_number = int(np.arange(len(all_variables))[np.array(all_variables) == variable])
    v_data = malaria_data['data'][d_number][:,v_number]
    #Close data file
    malaria_data.close()
    return v_data[:,None]

def ndvi_clean(district_name,variable_name):
    """
    Returns a variable in specific district form file uganda_ndvi_20130213.dat (weekly data) after removing wrong incidences
    """
    assert variable_name in ['ndvi','time','incidences']
    inc = ndvi_data(district_name,'incidences').flatten()
    var = ndvi_data(district_name,variable_name).flatten()
    index = np.arange(inc.size)
    mean = inc.mean()
    threshold = 2*inc.std()
    #Remove above
    clean = index[inc < mean+threshold]
    inc2 = inc[clean]
    var = var[clean]
    index2 = np.arange(clean.size)
    #Remove below
    clean2 = index2[inc2 > mean-threshold]
    var = var[clean2]
    """
    if district_name == 'Apac':
        clean = index[inc < 10000]
        var = var[clean,:]
    #Remove wrong data
    if district_name == 'Mbarara':
        v_data = np.hstack([v_data[:14],v_data[15:]])
    if district_name == 'Gulu':
        v_data = np.hstack([v_data[:-2],v_data[-1]])
    """
    return var[:,None]

#Incidences per location - Box diagrams
"""
var_list = ['altitude','latitude','longitude']
width_list = [30,120,120]
scale = [1,100,100]
for var,wd,scl in zip(var_list,width_list,scale):
    min_ = 100000
    max_ = 0
    pb.figure()
    for district in all_districts:
        incidences = ndvi_clean(district,'incidences')
        location = ndvi_data(district,var)[0,0]/scl
        pb.boxplot(incidences,positions=[location],widths=wd)
        if min_ > location:
            min_ = location
        if max_ < location:
            max_ = location
    pb.xlabel('%s' %var)
    pb.ylabel('incidences')
    minimax = (max_ - min_)*.1
    min_ = min_ - minimax
    max_ = max_ + minimax
    pb.xlim(min_,max_)
"""

#Variables vs time
"""
var_list=['ndvi','incidences']
subplots = (211,212)
name_list = ('NDVI','incidences')
for district in all_stations:
    time = ndvi_clean(district,'time')
    pb.figure()
    for var,subs,name in zip(var_list,subplots,name_list):
        fig = pb.subplot(subs)
        v = ndvi_clean(district,var)
        pb.plot(time,v,'k')
        pb.ylabel('%s' %name)
        fig.yaxis.set_major_locator(pb.MaxNLocator(3))
        fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_xaxis().set_visible(True)
    pb.xlabel('time (days)')
    pb.suptitle('%s' %district)
"""
#standardized Variables vs time
"""
lag = 0
for district in all_stations:
    altitude = ndvi_data(district,'altitude')[0,0]
    incidences = ndvi_clean(district,'incidences')
    incidences = (incidences - incidences.mean())/incidences.std()
    time = ndvi_clean(district,'time')
    fig = pb.figure()

    v_ = ndvi_clean(district,'ndvi')
    v = (v_ - v_.mean())/v_.std()
    pb.plot(time-lag,incidences,'r')
    pb.plot(time,v,'k--',linewidth=2)
    pb.ylabel('incidence / NDVI\nstandardized')
    #fig.yaxis.set_major_locator(pb.MaxNLocator(3))
    pb.text(2,pb.ylim()[1],'lag %s' %lag)
    pb.xlim(0,1800)
    pb.xlabel('time (days)')
    pb.suptitle('%s (altitude: %s)' %(district,altitude))

"""
#Forcast
Xlist_ = []
Xlist = []
Xlist_fut = []
Ylist = []
Ylist_ = []
Ylist_fut = []
likelihoods = []

stations = all_stations[:2]
I = np.arange(len(stations))
for i,district in zip(I,stations):
    #data
    Y_ = ndvi_clean(district,'incidences')
    X1_ = ndvi_clean(district,'time')

    #cut
    last = X1_[-1,0]
    cut = X1_[X1_ < last - 360].size

    Y = Y_[:cut,:]
    Y_fut = Y_[cut:,:]

    X1 = X1_[:cut,:]
    X1_fut = X1_[cut:,:]

    likelihoods.append(GPy.likelihoods.Gaussian(Y,normalize=False))
    Ylist_.append(Y_)
    Ylist.append(Y)
    Ylist_fut.append(Y_fut)

    #Index time
    Xlist_.append(np.hstack([np.repeat(i,X1_.size)[:,None],X1_]))
    Xlist.append(np.hstack([np.repeat(i,X1.size)[:,None],X1]))
    Xlist_fut.append(np.hstack([np.repeat(i,X1_fut.size)[:,None],X1_fut]))

#model 4
print '\nmodel 4'

R = 2
D = 1
periodic4 = GPy.kern.periodic_exponential(1)
rbf4 = GPy.kern.rbf(1)
linear4 = GPy.kern.linear(1)
bias4 = GPy.kern.bias(1)
base_white4 = GPy.kern.white(1)
white4 = GPy.kern.cor_white(base_white4,R,index=0,Dw=2) #FIXME
#base4 = linear4+periodic4*rbf4 + white4
#base4 = linear4+periodic4*rbf4 +bias4
base4 = rbf4.copy()+rbf4.copy()# + bias4
kernel4 = GPy.kern.icm(base4,R,index=0,Dw=2)

Z = np.linspace(100,1700,6)[:,None]

#m4 = GPy.models.mGP(Xlist, likelihoods, kernel4+white4, normalize_Y=True)
m4 = GPy.models.multioutput_GP(Xlist, likelihoods, kernel4+white4,Z=Z, normalize_X=True,normalize_Y=True)

m4.ensure_default_constraints()

m4.constrain_positive('kappa')
m4.constrain_fixed('iip',m4.Z.flatten())
#m4.set('exp_len',.1)
#m4.set('exp_var',10)
m4.set('1_len',.1)
m4.set('1_var',10)
#m4.set('rbf_var',.5)
m4.set('W',np.random.rand(R*2))

print m4.checkgrad(verbose=1)
m4.optimize()
print m4

fig=pb.subplot(233)
min2_ = X1_.min()
max2_ = X1_.max()
mean_,var_,lower_,upper_ = m4.predict(Xlist_[0])
GPy.util.plot.gpplot(Xlist_[0][:,1],mean_,lower_,upper_)
pb.plot(Xlist[0][:,1],Ylist[0],'kx',mew=1.5)
pb.plot(Xlist_fut[0][:,1],Ylist_fut[0],'rx',mew=1.5)
#pb.ylim(ym_lim)
pb.ylabel('incidences')
fig.xaxis.set_major_locator(pb.MaxNLocator(6))

fig=pb.subplot(236)
min2_ = X1_.min()
max2_ = X1_.max()
mean_,var_,lower_,upper_ = m4.predict(Xlist_[1])
GPy.util.plot.gpplot(Xlist_[1][:,1],mean_,lower_,upper_)
pb.plot(Xlist[1][:,1],Ylist[1],'kx',mew=1.5)
pb.plot(Xlist_fut[1][:,1],Ylist_fut[1],'rx',mew=1.5)
#pb.ylim(yw_lim)
#pb.ylabel('ndvi')
pb.xlabel('time (days)')
fig.xaxis.set_major_locator(pb.MaxNLocator(6))
