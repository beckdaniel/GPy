"""
Multioutput GP for malaria counts
dataset 1: ../../../playground/malaria/allDataWithoutWeather_20130125
dataset 2: ../../../playground/malaria/allWeatherStationData_20130211
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
Masindi <-                            Luwero
           Mpigi - Wakiso - Kampala - Mukono
"""

import numpy as np
import pylab as pb
import shelve
import GPy
import datetime
pb.ion()
pb.close('all')

def string2date(string): #NOTE Not used
#Converts a string into a date, this function is needed to format the dataset
    year = int(string[0:2])
    month = int(string[3:5])
    day = int(string[6:8])
    return 2000+year, month, day

def rm_dpl(index):
    used_terms = []
    return np.array([s for s in index if s not in used_terms and not used_terms.append(s)])

X_list = [] # Input list

data_wo_weather = []
# Read data
file_1='../../../playground/malaria/allDataWithoutWeather_20130125'
file_2='../../../playground/malaria/allWeatherStationData_20130211'
start_date = datetime.date(2003,01,05) # Reference date, i.e. time zero

df=shelve.open(file_1)
hk=shelve.open(file_2)

# Format allDataWithoutWeather_20130125
#districts = df.keys()
districts = ['Masindi','Mpigi','Wakiso','Kampala','Mukono','Luwero','Tororo']
date_inc = []
incidences = []
altitude = []
longitude = []
latitude = []
ndvi = []

year_ = []
month_ = []
day_ = []

for d,n in zip(districts,range(len(districts))):
    N = len(df[d]['malaria'].keys())
    new_date = []
    date_inc.append([])
    incidences.append([])
    years = df[d]['malaria'].keys()
    years.sort()
    ndvi_ =[]
    y_ = []
    m_ = []
    d_ = []
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

                    m_.append(month_i)
                    y_.append(year_i)
                    d_.append(day_i)
                    if day_i <= 10:
                        ndvi_.append(df[d]['ndvi'][year_i][month_i][0])
                    elif day_i <=20:
                        ndvi_.append(df[d]['ndvi'][year_i][month_i][1])
                    else:
                        ndvi_.append(df[d]['ndvi'][year_i][month_i][2])

    num_obs = len(incidences[-1])
    output_number = np.repeat(n,num_obs)[:,None] # This is the index column for the coreg_kernel
    incidences[-1] = np.array(incidences[-1])[:,None]
    date_inc[-1] = np.array(date_inc[-1])[:,None]
    altitude.append(np.repeat(df[d]['altitude'],num_obs)[:,None])
    longitude.append(np.repeat(df[d]['longitude'],num_obs)[:,None])
    latitude.append(np.repeat(df[d]['latitude'],num_obs)[:,None])
    ndvi.append(np.array(ndvi_)[:,None])
    X_list.append(np.hstack([output_number,date_inc[-1],longitude[-1],latitude[-1]]))
    data_wo_weather.append(np.hstack([incidences[-1],output_number,date_inc[-1],longitude[-1],latitude[-1],altitude[-1],ndvi[-1]]))
    year_.append(np.array(y_)[:,None])
    month_.append(np.array(m_)[:,None])
    day_.append(np.array(d_)[:,None])

# Format weather data
locations = hk.keys()
S_longitude = []
S_latitude = []
S_date = []
humid_06 = []
humid_12 = []
temp_max = []
temp_min = []
for loc in locations:
    S_longitude.append(hk[loc]['longitude'])
    S_latitude.append(hk[loc]['latitude'])
    S_date.append([])
    new_date = []
    h_06 = []
    h_12 = []
    t_min = []
    t_max = []
    humid_06.append([])
    humid_12.append([])
    temp_min.append([])
    temp_max.append([])
    years = hk[loc].keys()
    years.sort()
    years = years[:-2] #Remove longitude and latitude
    for year_i in years:
        months = hk[loc][year_i].keys()
        months.sort()
        for month_i in months:
            days = hk[loc][year_i].keys()
            days.sort()
            for day_i in days:
                new_date.append(datetime.date(int(year_i),int(month_i),int(day_i)))
                S_date[-1].append( (new_date[-1]-start_date).days )
                h_06.append(hk[loc][year_i][month_i][day_i]['humid_06'])
                if type(h_06[-1]) is unicode:
                    h_06[-1] = None
                h_12.append(hk[loc][year_i][month_i][day_i]['humid_12'])
                if type(h_12[-1]) is unicode:
                    h_12[-1] = None
                t_min.append(hk[loc][year_i][month_i][day_i]['temp_max'])
                t_max.append(hk[loc][year_i][month_i][day_i]['temp_min'])
    weeks = range(0,1716,7)
    h_06 = np.array(h_06)
    h_12 = np.array(h_12)
    t_min = np.array(t_min)
    t_max = np.array(t_max)
    for week_i in weeks:
         _range = np.arange(len(S_date[-1]))
         _tmp = np.array(S_date[-1])
         _range = _range[_tmp <= week_i]
         _tmp = _tmp[_range]
         _range = _range[_tmp > week_i-7]
         _tmp = _tmp[_range]

         # humid_06
         if 0 in np.array(h_06)[_range]:
             a = KJkj
         last_week = np.array(filter(None,np.array(h_06)[_range])) #Remove None elements
         if last_week.size > 3: #At least 4 days of the week
             humid_06[-1].append(last_week.mean())
         else:
             humid_06[-1].append(-100)

         # humid_12
         if 0 in np.array(h_12)[_range]:
             a = KJkj
         last_week = np.array(filter(None,np.array(h_12)[_range])) #Remove None elements
         if last_week.size > 3: #At least 4 days of the week
             humid_12[-1].append(last_week.mean())
         else:
             humid_12[-1].append(-100)

         # temp_min
         if 0 in np.array(t_min)[_range]:
             a = KJkj
         last_week = np.array(filter(None,np.array(t_min)[_range])) #Remove None elements
         if last_week.size > 3: #At least 4 days of the week
             temp_min[-1].append(last_week.mean())
         else:
             temp_min[-1].append(-100)

         # temp_max
         if 0 in np.array(t_max)[_range]:
             a = KJkj
         last_week = np.array(filter(None,np.array(t_max)[_range])) #Remove None elements
         if last_week.size > 3: #At least 4 days of the week
             temp_max[-1].append(last_week.mean())
         else:
             temp_max[-1].append(-100)

weather_weekly = []
weeks = np.array(weeks)[:,None]
for j in range(len(locations)):
    humid_06[j] = np.array(humid_06[j])[:,None]
    humid_12[j] = np.array(humid_12[j])[:,None]
    temp_min[j] = np.array(temp_min[j])[:,None]
    temp_max[j] = np.array(temp_max[j])[:,None]
    weather_weekly.append(np.hstack([weeks,humid_06[j],humid_12[j],temp_min[j],temp_max[j]]))

weather_final = []
for weather_i in weather_weekly:
    weather_final.append([])
    for wi in weather_i:
        if -100 not in wi:
            weather_final[-1].append(wi[None,:])
    weather_final[-1] = np.vstack(weather_final[-1])

data = []
stations_loc = np.hstack([np.array(S_longitude)[:,None],np.array(S_latitude)[:,None]])
_range = np.arange(stations_loc.shape[0])
for d,j in zip(districts,range(len(districts))):
    data.append([])
    district_loc = np.array([df[d]['longitude'],df[d]['latitude']])[None,:]
    distance = np.sqrt(np.sum((district_loc - stations_loc)**2,-1))
    min = distance.min()
    station = _range[distance == min]
    for row_i in weather_final[station]:
        if row_i[0] in data_wo_weather[j][:,2].flatten():
            tmp = np.arange(data_wo_weather[j][:,2].size)[data_wo_weather[j][:,2].flatten() == row_i[0]]
            data[-1].append(np.hstack([row_i,data_wo_weather[j][tmp,:].flatten()])) #cols 0 and 2 + 5
    data[-1] = np.vstack(data[-1])

"""
    years.sort()
    for year_i


#Set number of districts to work with (i.e. number of outputs)
R = len(districts)

# Define Gaussian likelihood
likelihoods = []
for y in incidences:
    likelihoods.append(GPy.likelihoods.Gaussian(y))

# Define the inducing inputs
M = R #NOTE: the model won't work properly if M is different from R
Z_index = [np.repeat(i,M)[:,None] for i in range(len(X_list))]
Z_date_inc = [np.linspace(0,1300,M)[:,None] for x in X_list ]
Z_longitude = [ x[:M,:] for x in longitude]
Z_latitude = [ x[:M,:] for x in latitude]
Z_list = [ np.hstack([a,b,c,d]) for a,b,c,d in zip(Z_index,Z_date_inc,Z_longitude,Z_latitude) ]

# Define coreg_kern and base kernels
rbf = GPy.kern.rbf(3)
bias = GPy.kern.bias(3)
noise = GPy.kern.white(3)
base = rbf + noise #+ bias
kernel = GPy.kern.icm(base,R,index=0,Dw=2)

# Define the model
m = GPy.models.multioutput_GP(X_list, likelihoods, kernel,Z_list, normalize_X=True) #NOTE: better to normalize X and Y

# Constraints
m.scale_factor = 1.
m.ensure_default_constraints()
m.unconstrain('rbf_var')
m.constrain_fixed('rbf_var',1.) # Variance parameter will be given by the elements of the coregionalization matrix
m.constrain_positive('kappa')
m.constrain_positive('W')
m.constrain_fixed('iip',m.Z[:,m.input_cols].flatten()) #No need to optimize this
m.set('len',.1) #NOTE the model works better initializing lengthscale as .1

# Optimize
print m.checkgrad(verbose=True)
m.optimize()

# Plots
#m.plot()
for r in range(R):
    pb.figure()
    m.plot_HD(input_col=1,output_num=r)
print m
print np.round(m.kern.parts[0].B,2)
"""
