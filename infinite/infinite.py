import math
import matplotlib.pylab as py
import numpy as np

class Scene:
	def __init__(self, ID, traits):
		(lam, B, alpha, t) = traits
		self.ID = ID
		self.lam = lam*1.0
		self.B = B*1.0
		self.alpha = alpha*1.0
		self.lamB = self.lam*self.B
		self.trueT = t #the number of periods I have control over

	def setDPTable(self, DPTable):
		#structure of DPTable:
		#DPT[t][i] =(x,y) means at time trueT-t, given rate (i+1)/N, 
		#we should go to rate x for rev y
		self.DPTable = DPTable
		
		#since I have the DPTable, might as well initialize a path dictionary
		self.pathDict={}

	def setVTable(self, V):
		self.V = V

	def setLTable(self, L):
		self.L = L


	def setOptAndCrit(self, rStar, pStar, rCrit):
		self.rStar = rStar
		self.pStar = pStar
		self.rCrit = rCrit #the critical point for price pStar


######################################
##functions for computing opt paths if given infinite horizon, control over T horizons

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


#################################33
#To compare one time to change vs two perdiods to change

#a binary searcher
#d= 1 means increasing function
#d= -1 means decreasing function
def binarySearch(d, l, u, goal, f):
	if d==1:
		while u-l >0.0000001:
			m = (l+u)/2.0
			val = f(m)

			if val > goal:
				u = m
			else:
				l=m

	else:
		while u-l > 0.0000001:
			m = (l+u)/2.0
			val = f(m)

			if val < goal:
				u = m
			else:
				l=m

	return (l+u)/2.0

#Find opt equil
#input:  x, a scene
def findOptPrice(x):
	l =math.exp(-1)
	u = 1.0

	rStar = binarySearch(1, l, u, x.lamB, lambda x: 2*x*math.log(x) + x)
	pStar = rStar/(2*x.lam) + x.B/2.0

	#since we are here, let's find the critical point for price pStar
	l= 0.0
	u = math.exp(-1.0)
	goal = -x.lam*(pStar - x.B)
	rCrit = binarySearch(-1, l, u, goal, lambda x: x*math.log(x))

	x.setOptAndCrit(rStar, pStar, rCrit)
 


#find opt at t=1 given t=0 and t=2
def optTwo(r, z, x):

	opt = math.exp((x.lamB - x.alpha*z*math.log(z))/r - 1.0)

	return opt

#find the revenue for a time period given the current rate, the future rate, and x
def rev(rNow, rNext, x):
	return rNext/x.lam * (x.lamB - rNow*math.log(rNext))


#given one scene and a starting point, we want to get to opt. 
# What is the diff between taking jumping there in one step and in two step
def compareOneTwo(r, x):
	findOptPrice(x)
	
	#for one time period
	revOne_1 = rev(r, x.rStar, x)  
	revOne_2 = rev(x.rStar, x.rStar, x)
	revOne = revOne_1 + x.alpha* revOne_2

	#Let y be the middle position
	y = optTwo(r, x.rStar, x)

	#for two time periods
	revTwo_1 = rev(r, y, x)
	revTwo_2 = rev(y, x.rStar, x)
	revTwo = revTwo_1 + revTwo_2

	return revOne/revTwo 
 




###############################################

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
def writeToFile(X,Z, name):
	f = open(name+'path.txt', 'w')
	for i in range(0,N):
		f.write(str(X[i])+",")
		for t in range(0,T-1):
			f.write(str(Z[t][i][0]) + ",")
		f.write("\n")
	f.close()


#analyze scene, do plots, etc
def analyzeScene(scene):
	#Plot V
	V=[]
	DPTable=[]

	#structure of DPTable:
	#DPT[t][i] =(x,y) means at time trueT-t, given rate (i+1)/N, we should go to rate x for rev y
	for t in range(0,T):
		if t ==0:	
			y=VInf(X, scene)
			V.append(y)
		
	#		maxR = maximum(X,y)
	#		print trait, maxR

		else:
			z = VNow(X, V[t-1], scene)
			DPTable.append(z)
			y = [x[1] for x in z]
			V.append(y)


	plot(X,V, "Scene"+str(scene.ID)+"_V_T", [0,1,-0.2, 5])

	#writeToFile(X,DPTable,"Scene"+str(scene.ID))

	#plot L
	L = []
	for t in range(0,T):
		y=[V[t][j] - maintain(X[j], scene) for j in range(0,N)]
		L.append(y)

	#	if t !=0:
	#		minR = minimum(X,y)
		#	print trait, t, minR

	plot(X,L,"Scene"+str(scene.ID)+"_L_T", [0,1, -0.2, 5.0])

	scene.setDPTable(DPTable)
	scene.setVTable(V)
	scene.setLTable(L)

#make sure start is on the grid?
def extractPath(scene, start):
	p=[start]
	index = int(start*N-1)

	for i in range(T-2, -1, -1):
		nextRate = scene.DPTable[i][index][0]
		p.append(nextRate)
		index = int(nextRate*N-1)
	return p
	


#plot the paths for a bunch of different starters onto the same graph
def plotPath(scene):
	fig = py.figure()
	py.title("Scene"+str(scene.ID)+" path")
	py.xlabel("time")
	py.ylabel("rate")
	for x in scene.pathDict:
		py.plot(scene.pathDict[x], label = str(x))
 	py.axis([0,10,0,1])	
	fig.savefig("Scene"+str(scene.ID)+"_path")
	py.close(fig) 


#=======================================================
#grid size; will use r =1/N, 2/N, ..., 1
N=2000
X = [(i+1.0)/N for i in range(0,N)] 


#define a set of "starter" rates that we will consider
starter = [0.001, 0.01, 0.1, 0.2, 0.3,  0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#####################################################3
#define scenes
sceneDict={}

#traits(lam, B, alpha, T)
trait = (0.5, -1/3.0, 0.95, 20)
#sceneDict[1] = Scene(1, trait)

trait = (0.5, -1/3.0, 0.9, 20)
#sceneDict[2] = Scene(2, trait)

trait = (0.5, -1/3.0, 0.85, 20)
#sceneDict[3] = Scene(3, trait)

trait = (0.5, -1/3.0, 0.95, 3)
#sceneDict[4] = Scene(4, trait)

trait = (0.5, -1/3.0, 0.95, 2)
#sceneDict[5] = Scene(5, trait)

trait = (0.5, -1/3.0, 0.95, 1)
sceneDict[6] = Scene(6, trait)

trait = (1.0, -1/4.0, 0.95, 20)
#sceneDict[20] = Scene(20, trait)



###########################################

for i in sceneDict:
	myScene = sceneDict[i]
	T = myScene.trueT+1
	analyzeScene(myScene)
	for x in starter:
		path = extractPath(myScene, x)
		myScene.pathDict[x] = path
	
	plotPath(myScene)

	
#myScene = sceneDict[20]
#analyzeScene(myScene)
#print extractPath(myScene, 0.001)
