#NOTE BROKEN
"""
Multioutput GP for malaria counts
---------------------------------
dataset: ../../../playground/malaria/uganda_data_20130213.dat
B matrix controls the relation between districts
Incidences are assumed to have a log-normal distribution
"""

import numpy as np
import pylab as pb
import shelve
import GPy
import datetime
pb.ion()
pb.close('all')

#Load data
malaria_data = shelve.open('../../../playground/malaria/uganda_data_20130213.dat',writeback=False)
all_districts = malaria_data['districts']
all_variables = malaria_data['headers']
all_stations = malaria_data['stations']
station_variables = malaria_data['headers_daily']

district = ['Kampala']
d_number = int(np.arange(len(all_districts))[np.array(all_districts) == district])
time_number = int(np.arange(len(all_variables))[np.array(all_variables) == 'time'])
time = malaria_data['data'][d_number][:,time_number][:,None]
Y_variable = ['incidences']
Y_number = int(np.arange(len(all_variables))[np.array(all_variables) == Y_variable])
inc = malaria_data['data'][d_number][:,Y_number][:,None]
W_variable = ['ndvi']
W_number = int(np.arange(len(all_variables))[np.array(all_variables) == W_variable])
weather = malaria_data['data'][d_number][:,W_number][:,None]
W_variable2 = ['rain']
W_number2 = int(np.arange(len(all_variables))[np.array(all_variables) == W_variable2])
weather2 = malaria_data['data'][d_number][:,W_number2][:,None]

#Close data file
malaria_data.close()

inc_z = (inc - inc.mean())/inc.std()
weather_z = (weather - weather.mean())/weather.std()

#inc_diff = np.diff(inc_z.flatten())[:,None]
#weather_diff = np.diff(weather_z.flatten())[:,None]
inc_diff = np.log(inc[1:,:]/inc[:-1,:])
weather_diff = np.log(weather[1:,:]/weather[:-1,:])
timed1 = time[1:,:]

pb.subplot(211)
pb.plot(time,weather_z,'b') #+30
pb.xlim(0,2000)
#pb.subplot(412)
pb.plot(time,inc_z,'r')
pb.xlim(0,2000)
pb.subplot(212)
pb.plot(timed1,weather_diff,'b') #+30
pb.xlim(0,2000)
#pb.subplot(414)
pb.plot(timed1,inc_diff,'r')
pb.xlim(0,2000)

"""
Model for weather variable, weather vs time
"""
print 'Weather model'
#kernel
periodic_w = GPy.kern.periodic_exponential(1)
rbf_w = GPy.kern.rbf(1)
linear_w = GPy.kern.linear(1)
white_w = GPy.kern.white(1)
#likelihood
like_w = GPy.likelihoods.Gaussian(weather,normalize =True)
#model
q = GPy.models.GP(time, like_w, linear_w+periodic_w+rbf_w+white_w, normalize_X=True)
#optimize
q.ensure_default_constraints()
print q.checkgrad()
q.optimize()
print q
#plot
pb.figure()
q.plot()
#prediction
time_star = np.arange(1500,3000)[:,None]
mean_w,var_w,lower_w,upper_w = q.predict(time_star)
GPy.util.plot.gpplot(time_star,mean_w,lower_w,upper_w)
pb.xlim(0,3000)
pb.title('Weather variable')

"""
Model for weather variable 2, weather vs time
"""
print 'Weather model 2'
#kernel
periodic_w2 = GPy.kern.periodic_exponential(1)
linear_w2 = GPy.kern.linear(1)
rbf_w2 = GPy.kern.rbf(1)
white_w2 = GPy.kern.white(1)
#likelihood
like_w2 = GPy.likelihoods.Gaussian(weather2,normalize =True)
#model
q2 = GPy.models.GP(time, like_w2, linear_w2+periodic_w2+rbf_w2+white_w2, normalize_X=True)
#optimize
q2.ensure_default_constraints()
print q2.checkgrad()
q2.optimize()
print q2
#plot
pb.figure()
q2.plot()
#prediction
mean_w2,var_w2,lower_w2,upper_w2 = q2.predict(time_star)
GPy.util.plot.gpplot(time_star,mean_w2,lower_w2,upper_w2)
pb.xlim(0,3000)
pb.title('Weather variable 2')


"""
Model 1 for incidence, incidence vs time
"""
print 'Incidence model 1'
#kernel
periodic_1 = GPy.kern.periodic_exponential(1)
rbf_1 = GPy.kern.rbf(1)
linear_1 = GPy.kern.linear(1)
white_1 = GPy.kern.white(1)
#likelihood
like_1 = GPy.likelihoods.Gaussian(inc,normalize =True)
#model
m_1 = GPy.models.GP(time, like_1,linear_1+ periodic_1+rbf_1+white_1, normalize_X=True)
#optimize
m_1.ensure_default_constraints()
print m_1.checkgrad()
m_1.optimize()
print m_1
#plot
pb.figure()
m_1.plot()
#prediction
mean_1,var_1,lower_1,upper_1 = m_1.predict(time_star)
GPy.util.plot.gpplot(time_star,mean_1,lower_1,upper_1)
pb.xlim(0,3000)
pb.title('Incidence - time')
#pb.ylim(-1000,6000)

"""
Model 2 for incidence, Poisson incidence vs time
"""
"""
print 'Incidence model Poisson'
#kernel
periodic_2 = GPy.kern.periodic_exponential(1)
rbf_2 = GPy.kern.rbf(1)
noise_2 = GPy.kern.white(1)
#likelihood
distribution_2 = GPy.likelihoods.likelihood_functions.Poisson()
like_2 = GPy.likelihoods.EP(inc,distribution_2)
#model
m_2 = GPy.models.GP(time, like_2, periodic_2+rbf_2, normalize_X=False)
#optimize
m_2.update_likelihood_approximation()
m_2.ensure_default_constraints()
#m_2.set('exp_var',.1)
#m_2.set('rbf_var',10)
print m_2.checkgrad()
m_2.optimize()
print m_2
#plot
pb.figure()
m_2.plot()
#prediction
mean_2,var_2,lower_2,upper_2 = m_2.predict(time_star)
GPy.util.plot.gpplot(time_star,mean_2,lower_2,upper_2)
pb.xlim(0,3000)
pb.title('Poisson Incidence - time')
"""
"""
Model 3 for incidence
"""
print 'Incidence model <- weather'
#kernel
periodic_3 = GPy.kern.periodic_exponential(1)
rbf_3 = GPy.kern.rbf(2)
linear_3 = GPy.kern.linear(1)
bias_3 = GPy.kern.bias(2)
white_3 = GPy.kern.white(2)
#likelihood
like_3 = GPy.likelihoods.Gaussian(inc,normalize =True)
#model
#m_3 = GPy.models.GP(np.hstack([time,weather]), like_3,linear_3*periodic_3+linear_3.copy()*rbf_3+white_3, normalize_X=True)
m_3 = GPy.models.GP(np.hstack([time,weather]), like_3,linear_3.copy()*periodic_3+white_3, normalize_X=True)
#optimize
m_3.ensure_default_constraints()
#m_2.set('exp_len',1)
print m_3.checkgrad(verbose=True)
m_3.optimize()
print m_3
#prediction
pb.figure()
mean_3_1,var_3_1,lower_3_1,upper_3_1 = m_3.predict(np.hstack([time,weather]))
GPy.util.plot.gpplot(time,mean_3_1,lower_3_1,upper_3_1)
pb.plot(time,inc,'kx',mew=1.5)
mean_3,var_3,lower_3,upper_3 = m_3.predict(np.hstack([time_star,mean_w]))
GPy.util.plot.gpplot(time_star,mean_3,lower_3,upper_3)
pb.xlim(0,3000)
pb.title('Incidence - time+weather')
#pb.ylim(-1000,6000)

"""
Model 4 for incidence
"""
"""
print 'Incidence model 1'
#kernel
periodic_4 = GPy.kern.periodic_exponential(1)
linear_4 = GPy.kern.linear(1)
bias_4 = GPy.kern.bias(3)
rbf_4 = GPy.kern.rbf(2)
noise_4 = GPy.kern.white(3)
#likelihood
like_4 = GPy.likelihoods.Gaussian(inc,normalize =True)
#model
#m_4 = GPy.models.GP(np.hstack([time,weather,weather2]), like_4, linear_4+rbf_4, normalize_X=True)
m_4 = GPy.models.GP(np.hstack([time,weather,weather2]), like_4, linear_4*rbf_4+noise_4, normalize_X=True)
#optimize
m_4.ensure_default_constraints()
#m_4.set('2_len',1)
#m_4.set('1_len',.1)
print m_4.checkgrad(verbose=True)
m_4.optimize()
print m_4
#prediction
pb.figure()
mean_4_1,var_4_1,lower_4_1,upper_4_1 = m_4.predict(np.hstack([time,weather,weather2]))
GPy.util.plot.gpplot(time,mean_4_1,lower_4_1,upper_4_1)
pb.plot(time,inc,'kx',mew=1.5)
mean_4,var_4,lower_4,upper_4 = m_4.predict(np.hstack([time_star,mean_w,mean_w2]))
GPy.util.plot.gpplot(time_star,mean_4,lower_4,upper_4)
pb.xlim(0,3000)
pb.title('Incidence - time+weather+weather')
#pb.ylim(-1000,6000)
"""
