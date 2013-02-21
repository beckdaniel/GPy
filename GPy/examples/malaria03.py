"""
Multioutput GP for malaria counts
dataset: ../../../playground/malaria/allDataWithoutWeather_20130125
data structure:
- district:
    - malaria:
        - year:
            - month:
                - week -> number of sick people
    - area
    - altitude
    - longitude
    - latitude
    - ndvi:
        - 10 days image


The model is applied to  Districts that are geographicaly close to each other
W is a 2 x R matrix
---------------------------------------------
Masindi <-                            Luweero
           Mpigi - Wakiso - Kampala - Mukono
"""
#NOTE this test just considers incidences

import numpy as np
import pylab as pb
import shelve
import GPy
import datetime
pb.ion()
pb.close('all')

def string2date(string):
#Converts a string into a date, this function is needed to format the dataset
    year = int(string[0:2])
    month = int(string[3:5])
    day = int(string[6:8])
    return 2000+year, month, day

# Process Malaria data
filename='../../../playground/malaria/allDataWithoutWeather_20130125'
df=shelve.open(filename)
#districts = df.keys()
districts = ['Masindi','Mpigi','Wakiso','Kampala','Mukono','Luwero']

# Format dates and number of incidences
date_inc = []
incidences = []
start_date = datetime.date(2003,01,01) # Reference date, i.e. time zero
for d,n in zip(districts,range(len(districts))):
    N = len(df[d]['malaria'].keys())
    new_date = []
    date_inc.append([])
    incidences.append([])
    years = df[d]['malaria'].keys()
    years.sort()
    for year_i in years:
        months = df[d]['malaria'][year_i].keys()
        months.sort()
        for month_i in months:
            days = df[d]['malaria'][year_i][month_i].keys()
            days.sort()
            for day_i in days:
                incidence_i = df[d]['malaria'][year_i][month_i][day_i]
                if incidence_i is not None:
                    new_date.append(datetime.date(int(year_i),int(month_i),int(day_i)))
                    date_inc[-1].append((new_date[-1] - start_date).days)
                    incidences[-1].append(float(incidence_i))

    incidences[-1] = np.array(incidences[-1])[:,None]
    date_inc[-1] = np.array(date_inc[-1])[:,None]
    output_number = np.repeat(n,date_inc[-1].size)[:,None] # This is the index column for the coreg_kernel
    date_inc[-1] = np.hstack([output_number,date_inc[-1]])

#Set number of districts to work with (i.e. number of outputs)
R = len(districts)

# Define Gaussian likelihood
likelihoods = []
for y in incidences:
    likelihoods.append(GPy.likelihoods.Gaussian(y))

# Define the inducing inputs
M = R #NOTE: the model won't work properly if M is different from R
Zindex = [np.repeat(i,M)[:,None] for i in range(len(date_inc))]
_Z = [np.linspace(0,1300,M)[:,None] for a in range(M)]
Z_date_inc = [np.hstack([_i,_Z]) for _i,_Z in zip(Zindex,_Z)]

# Define coreg_kern and base kernels
rbf = GPy.kern.rbf(1)
bias = GPy.kern.bias(1)
noise = GPy.kern.white(1)
base = rbf + noise #+ bias
kernel = GPy.kern.icm(base,R,index=0,Dw=2)

# Define the model
m = GPy.models.multioutput_GP(X_list=date_inc,likelihood_list=likelihoods,kernel=kernel,Z_list=Z_date_inc,normalize_X=True) #NOTE: better to normalize X and Y

# Constraints
m.scale_factor = 1.
m.ensure_default_constraints()
m.unconstrain('rbf_var')
m.constrain_fixed('rbf_var',1.) # Variance parameter will be given by the elements of the coregionalization matrix
m.constrain_positive('kappa')
#m.constrain_positive('W')
m.constrain_fixed('iip',m.Z[:,m.input_cols].flatten()) #No need to optimize this
m.set('len',.1) #NOTE the model works better initializing lengthscale as .1
m.set('W',np.random.rand(0,1,2*R))
# Optimize
print m.checkgrad(verbose=True)
m.optimize()

# Plots
m.plot()
print m
print np.round(m.kern.parts[0].B,2)
