import numpy as np
f = file('log.txt','r')

data = f.read().split("====")[1:]

xs = []
Ps = []
ts = []

for s in data:
    exec s
    xs.append(np.array(x))
    Ps.append(np.array(P))
    ts.append(time)


from pylab import *
plot(sin(np.array(ts)),sin(2*np.array(ts)),'o')
plot(sin(np.array(3)),sin(2*np.array(3)),'or')

figure()
semilogy([x[0,0] for x in Ps])

figure()
plot([x[0] for x in xs])


show()
