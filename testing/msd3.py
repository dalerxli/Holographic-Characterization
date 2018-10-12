
import json
import numpy as np
from matplotlib import pyplot as plt

with open('sediment_predictions.json', 'r') as rfile:
    data = json.load(rfile)

x = np.array(data['x'])
y = np.array(data['y'])
z = np.array(data['z'])
a = np.array(data['a'])
n = np.array(data['n'])

#data is list, nlags is integer
#nlags is MAX timestep to compute
def msd3(data, nlags):
    totnpts = len(data)
    m = np.zeros((2, nlags))
    for i in range(nlags):
        tau = i+1
        d = data[tau:] - data[:-tau]
        nsets = tau
        npts = int(np.floor(len(d)/nsets))
        d = d[:(npts*nsets)]
        d = d.reshape(nsets, npts)
        d = np.multiply(d,d)
        if npts>1:
            mu = np.sum(d, axis=1)/npts
        else:
            mu=d[:]
        m[0][tau-1] = np.sum(mu)/nsets
        if npts>1:
            for j in range(nsets):
                d[j][:] -= mu[j]
            sigma = np.matmul(d, d.transpose())
            sigma /= npts
            sd = np.multiply(d,d)
            sd = np.sum(sd, axis=1)/npts
            sd = np.sqrt(np.matmul(sd, sd.transpose()))
            devsq = np.multiply(sigma, sigma) / sd / npts
            m[1][tau-1] = np.sqrt(np.sum(devsq)/nsets**2)
    return m


'''
time = np.linspace(0, len(x)-1, num=len(x))
plt.plot(time,x)
plt.show()
'''

[msd_x, dev_x] = msd3(x, 20)
[msd_y, dev_y] = msd3(y, 20)
tau = np.linspace(1, 20, num=20)


xpoly  = np.polyfit(tau,msd_x,1)
p_x = np.poly1d(xpoly)
polyfitx = p_x(tau)

ypoly = np.polyfit(tau,msd_y,1)
p_y = np.poly1d(ypoly)
polyfity = p_y(tau)

x_D = xpoly[0]/2
x_D = round(x_D, 2)
x_err = xpoly[1]
x_err /= 2
x_err = np.sqrt(x_err)
x_err = round(x_err, 2)

y_D = ypoly[0]/2
y_D = round(y_D, 2)
y_err = ypoly[1]
y_err /= 2
y_err = np.sqrt(y_err)
y_err = round(y_err, 2)



plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{tgheros}',
                                       r'\usepackage{sansmath}',
                                       r'\sansmath',
                                       r'\usepackage{siunitx}',
                                       r'\sisetup{detect-all}']

ticklabelsize = 20
axislabelsize = 30


fig, ax = plt.subplots(figsize=(7,7))
ax.errorbar(tau, msd_x, yerr=dev_x, fmt='bo', markersize=4, capsize=2)
ax.plot(tau, polyfitx, 'r')
ax.set_xlabel(r'$\tau$', fontsize=axislabelsize)
ax.set_ylabel(r'$msd$', fontsize=axislabelsize)
plt.text(3,250,r'$\epsilon$=%s'%x_err, bbox=dict(facecolor='white',alpha=1), fontsize=axislabelsize)
ax.tick_params(labelsize=ticklabelsize)

# grid under transparent plot symbols
ax.set_axisbelow(True)
ax.grid(color='k', linestyle='dotted', lw=1, alpha=0.5)
#plt.show()

plt.savefig('../plots/x_msd.png', bbox_inches='tight')


fig, ax = plt.subplots(figsize=(7,7))
ax.errorbar(tau, msd_y, yerr=dev_y, fmt='bo', markersize=4, capsize=2)
ax.plot(tau, polyfity, 'r')
ax.set_xlabel(r'$\tau$', fontsize=axislabelsize)
ax.set_ylabel(r'$msd$', fontsize=axislabelsize)
plt.text(3,250,r'$\epsilon$=%s'%y_err, bbox=dict(facecolor='white',alpha=1), fontsize=axislabelsize)
ax.tick_params(labelsize=ticklabelsize)

# grid under transparent plot symbols
ax.set_axisbelow(True)
ax.grid(color='k', linestyle='dotted', lw=1, alpha=0.5)
#plt.show()

plt.savefig('../plots/y_msd.png', bbox_inches='tight')


'''
msd_z = []
for i in range(20):
    msd_z.append(greedy_msd(z, (i+1)))
msd_z = np.array(msd_z)
msd_z = msd_z.reshape(len(msd_z))

zpoly = np.polyfit(tau,msd_z,1)
p_z = np.poly1d(zpoly)
polyfitz = p_z(tau)


plt.plot(tau, msd_z, 'bo')
plt.plot(tau, polyfitz, 'r')
plt.show()

print(zpoly)
'''
