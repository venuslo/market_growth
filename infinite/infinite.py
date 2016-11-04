import math
import matplotlib.pylab as py
import numpy as np

class Scene:
	def __init__(self, traits):
		(lam, B, alpha) = traits
		self.lam = lam*1.0
		self.B = B*1.0
		self.alpha = alpha*1.0
		self.lamB = self.lam*self.B

#compute the max of this period based on a grid
def Vr(r, X, vNext, scene):
	values = [(X[j]/scene.lam)*(scene.lamB - r*math.log(X[j])) \
			+ scene.alpha *vNext[j] for j in range(0,N)]

	return values

def VInf(X, scene):
	
	values = [maintain(x, scene) for x in X]
	return values

def maintain(x, scene):
	return (x/scene.lam)*(scene.lamB - x*math.log(x)) * 1.0/(1-scene.alpha)

def VNow(X, vNext, scene):
	Y = [Vr(r, X, vNext, scene) for r in X]
	vNow = [max(y) for y in Y]
	return vNow


def plot(X, Y, name, axes):
	for t in range(0,T):
		
		fig = py.figure()
		py.title(name + str(T-t))
		py.xlabel("given rate")
		py.ylabel("discounted profit")
		py.axis(axes)
		py.plot(X, Y[t])
		fig.savefig(name+str(T-t))
		py.close(fig) 


#=======================================================
#grid size; will use r =1/N, 2/N, ..., 1
N=100

X = [(i+1.0)/N for i in range(0,N)] 

#traits(lam, B, alpha)
trait = (1.0, -1/4.0, 0.95)
scene1 = Scene(trait)

#number of time periods to go back
T = 5

#Plot V
V=[]
for t in range(0,T):
	if t ==0:
		V.append(VInf(X, scene1))

	else:
		y = VNow(X, V[t-1], scene1)
		V.append(y)


plot(X,V, "Scene1_V_T", [0,1,-0.2, 1.5])


#plot L
L = []
for t in range(0,T):
	y=[V[t][j] - maintain(X[j], scene1) for j in range(0,N)]
	L.append(y)

plot(X,L,"Scene1_L_T", [0,1, -0.2, 5.0])
