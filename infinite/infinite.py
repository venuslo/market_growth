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
#returns vector
def Vr(r, X, vNext, scene):
	values = [(X[j]/scene.lam)*(scene.lamB - r*math.log(X[j])) \
			+ scene.alpha *vNext[j] for j in range(0,N)]

	return values

def VInf(X, scene):
	
	values = [maintain(x, scene) for x in X]
	return values

def maintain(x, scene):
	return (x/scene.lam)*(scene.lamB - x*math.log(x)) * 1.0/(1-scene.alpha)

#inpute:  vector, vector, obj
#return: vector of tuples
def VNow(X, vNext, scene):
	Y = [Vr(r, X, vNext, scene) for r in X]
	vNow = [maximum(X,y) for y in Y] #returns the rate and value
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



def minimum(X,Y):
	minX = X[0]
	minY = Y[0]

	for i in range(0, len(X)):
		if Y[i] < minY:
			minX = X[i]
			minY = Y[i]

	return (minX, minY)

def maximum(X,Y):
	maxX = X[0]
	maxY = Y[0]

	for i in range(0,len(X)): 
		if Y[i]> maxY:
			maxX = X[i]
			maxY = Y[i]

	return (maxX, maxY)


#input:  X a vector, Z a matrix
def writeToFile(X,Z):
	f = open('path.txt', 'w')
	for i in range(0,N):
		f.write(str(X[i])+",")
		for t in range(0,T-1):
			f.write(str(Z[t][i][0]) + ",")
		f.write("\n")
	f.close()


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
Z=[]
for t in range(0,T):
	if t ==0:	
		y=VInf(X, scene1)
		V.append(y)
		
		maxR = maximum(X,y)
		print trait, maxR

	else:
		z = VNow(X, V[t-1], scene1)
		Z.append(z)
		y = [x[1] for x in z]
		V.append(y)


plot(X,V, "Scene1_V_T", [0,1,-0.2, 1.5])

writeToFile(X,Z)

#plot L
L = []
for t in range(0,T):
	y=[V[t][j] - maintain(X[j], scene1) for j in range(0,N)]
	L.append(y)

	if t !=0:
		minR = minimum(X,y)
		print trait, t, minR

plot(X,L,"Scene1_L_T", [0,1, -0.2, 5.0])
