import math
import numpy as np
import matplotlib as mpl

bigdble=1.E98

Rsun=7.E10
mH=1.67E-24 # mass of a H atom in CGS
G=6.674E-8
Msun=2.E33

R1=20*Rsun # specified by Jon in his simulations

a=[1.62,1.99] # orbital separation (rstar)
M2=[1.3,2.] # mass of the CO (msun)
# Those ones are from the out file of get_BC.py :
rho0=[7.23806989802e-15,3.62066199659e-15] # cgs
vinf=[925.E5,1141.E5] # cgs

a=np.asarray(a)
a=a*R1
M2=np.asarray(M2)
M2=M2*Msun
rho0=np.asarray(rho0)
vinf=np.asarray(vinf)

correct_for_mass=False # ** user-specified **
if (correct_for_mass):
    Gm_Edd_Jon=0.42
    M1_Jon    =50*Msun
    Gm_Edd =0.2 # ** user-specified ** Eddington factor
    M1     =25.*Msun # ** user-specified ** mass for the donor star
    print 'Dividing speeds by a factor : ', np.sqrt((M1_Jon*(1.-Gm_Edd_Jon))/(M1*(1.-Gm_Edd)))
    vinf=vinf/np.sqrt((M1_Jon*(1.-Gm_Edd_Jon))/(M1*(1.-Gm_Edd)))
else:
    M1=25.*Msun # ** user-specified ** mass for the donor star

Racc=(2*G*M2)/vinf**2.
t0=Racc/vinf

Mloss=1.3E-6*(Msun/(3600.*24*365.24)) # stellar mass loss rate
tstart=[110.001,135.]# when do we start to use the mdot data
#tend  =160.# when do we stop

# duration1=175.*t0[0]
# duration2=183.*t0[1]
# Nbins=[200,int(200*(duration2/duration1))]

# Needed afterwhile, for NH

NHstat=True
NH0_ref=1.E43
beta_all=[0.,np.arcsin(R1/a),math.pi/4.,math.pi/2.]
beta_names=[r'0 (edge-on)','grazing',r'$\pi/4$',r'$\pi/2$ (face-on)']

# To bring back the NH measured in VAC in CGS
NH_scale=(rho0*((2*G*M2)/vinf**2.))/mH
time_scale=(2.*G*M2)/vinf**3.

P=2.*math.pi*np.sqrt(a**3./(G*(M1+M2)))
Rout=8*Racc #a/20. # outer simulation radius, 8Racc

# Infos on Jon's results, in CGS
folder='../04-Jon_insights/data_HR/'
Nr=1000 # total # of radial cells in Jon's simus
Ny=100 # total # of transverse cells in Jon's simus

input_file='../00-Streamlines_integration/output/info_safe_4' # contains all the mdot computed in 00-Streamlines
tmp_folder   ='tmp/'
outputs='output/'

figsize=8
fontsize=18
colors=['#8B0000','#006400','#00008B','#F08080','#00FF00','#87CEFA']
linestyles=['solid','dashed']
mark=["s","o","^"]
mpl.rc('font',size=fontsize,weight='bold')

typeMean='arithmetic' # ** user-specified ** type of mean (arithmetic, geometric or harmonic)
