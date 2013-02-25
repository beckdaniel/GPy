"""
Complementary part of malaria17
"""

#GP regressions
for district,nd in zip(all_stations,range(R)):
    #data
    Y_ = useful.ndvi_clean(district,'incidences')
    X1_ = useful.ndvi_clean(district,'time')
    X2_ = useful.ndvi_clean(district,'ndvi')

    #cut
    last = X1_[-1,0]
    cut = X1_[X1_ < last - 360].size

    Y = Y_[:cut,:]
    Y_fut = Y_[cut:,:]

    X1 = X1_[:cut,:]
    X1_fut = X1_[cut:,:]

    X2 = X2_[:cut,:]
    X2_fut = X2_[cut:,:]

    XX_ = np.hstack([X1_,X2_])
    XX = np.hstack([X1,X2])
    XX_fut = np.hstack([X1_fut,X2_fut])

    pb.figure()
    pb.suptitle('%s' %district)
    print '\n', district

    #weather 1
    print '\n', 'ndvi'
    likelihoodw1 = GPy.likelihoods.Gaussian(X2,normalize =True)

    periodicw1 = GPy.kern.periodic_exponential(1)
    rbfw1 = GPy.kern.rbf(1)
    biasw1 = GPy.kern.bias(1)
    linearw1 = GPy.kern.linear(1)
    whitew1 = GPy.kern.white(1)

    w1 = GPy.models.GP(X1, likelihoodw1, periodicw1*rbfw1+rbfw1.copy()+biasw1+whitew1, normalize_X=True)

    #w1.ensure_default_constraints() #NOTE not working for sum of rbf's
    w1.constrain_positive('var')
    w1.constrain_positive('len')
    print w1.checkgrad()
    w1.set('exp_len',.1)
    w1.set('exp_var',10)
    w1.set('rbf_var',.5)
    w1.optimize()
    print w1

    fig=pb.subplot(223)
    min1_ = X1_.min()
    max1_ = X1_.max()
    X1_star = np.linspace(min1_,max1_,200)[:,None]
    mean_,var_,lower_,upper_ = w1.predict(X1_star)
    GPy.util.plot.gpplot(X1_star,mean_,lower_,upper_)
    pb.plot(X1,X2,'kx',mew=1.5)
    pb.plot(X1_fut,X2_fut,'rx',mew=1.5)
    pb.ylabel('ndvi')
    #pb.xlabel('time (days)')
    pb.suptitle('%s' %district)
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

    #trends comparison
    """
    fig=pb.subplot(224)
    Yz_ = (Y_-Y_.mean())/Y_.std()
    X2z_ = (X2_-X2_.mean())/X2_.std()
    pb.plot(X1_,Yz_,'b')
    pb.plot(X1_,X2z_,'k--',linewidth=1.5)
    pb.ylabel('Incidence / ndvi\n(standardized)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))
    """

    #model 1
    print '\nmodel 1'
    likelihood1 = GPy.likelihoods.Gaussian(Y,normalize =True)

    periodic1 = GPy.kern.periodic_exponential(1)
    rbf1 = GPy.kern.rbf(1)
    linear1 = GPy.kern.linear(1)
    white1 = GPy.kern.white(1)

    m1 = GPy.models.GP(X1, likelihood1, rbf1*periodic1+rbf1.copy()+white1, normalize_X=True)

    #m1.ensure_default_constraints() #NOTE not working for sum of rbf's
    m1.constrain_positive('var')
    m1.constrain_positive('len')
    m1.set('exp_len',.1)
    m1.set('exp_var',10)
    m1.set('rbf_var',.5)
    print m1.checkgrad()
    m1.optimize()
    print m1

    #pb.figure()
    fig=pb.subplot(221)
    min1_ = X1_.min()
    max1_ = X1_.max()
    mean_,var_,lower_,upper_ = m1.predict(X1_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,Y,'kx',mew=1.5)
    pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
    pb.ylabel('incidences')
    ylim=pb.ylim()
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))

    #model 2
    print '\nmodel 2'
    likelihood2 = GPy.likelihoods.Gaussian(Y,normalize =True)

    periodic2 = GPy.kern.periodic_exponential(1)
    rbf2 = GPy.kern.rbf(1)
    rbf2_ = GPy.kern.rbf(2)
    linear2 = GPy.kern.linear(1)
    white2 = GPy.kern.white(2)

    m2 = GPy.models.GP(XX, likelihood2, GPy.kern.kern.prod_orthogonal(rbf2,periodic2)+rbf2_+white2, normalize_X=True)

    #m2.ensure_default_constraints() #NOTE not working for sum of rbf's
    m2.constrain_positive('var')
    m2.constrain_positive('len')
    m2.set('exp_len',.1)
    m2.set('exp_var',10)
    m2.set('rbf_var',.5)
    print m2.checkgrad()
    m2.optimize()
    print m2

    fig=pb.subplot(224)
    min2_ = X1_.min()
    max2_ = X1_.max()
    mean_,var_,lower_,upper_ = m2.predict(XX_)
    GPy.util.plot.gpplot(X1_,mean_,lower_,upper_)
    pb.plot(X1,Y,'kx',mew=1.5)
    pb.plot(X1_fut,Y_fut,'rx',mew=1.5)
    pb.ylim(ylim)
    pb.ylabel('incidences')
    pb.xlabel('time (days)')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))


    #model 4 plots
    fig=pb.subplot(222)
    mean_,var_,lower_,upper_ = m4.predict(Xlist_[nd])
    GPy.util.plot.gpplot(Xlist_[nd][:,1],mean_,lower_,upper_)
    pb.plot(Xlist[nd][:,1],Ylist[nd],'kx',mew=1.5)
    pb.plot(Xlist_fut[nd][:,1],Ylist_fut[nd],'rx',mew=1.5)
    if hasattr(m4,'Z'):
        _Z = m4.Z[:m4._M,1]*m4._Zstd[:,1]+m4._Zmean[:,1]
        pb.plot(_Z,np.repeat(pb.ylim()[0],m4._M),'r|',mew=1.5)
    pb.ylim(ym_lim)
    pb.ylabel('incidences')
    fig.xaxis.set_major_locator(pb.MaxNLocator(6))
