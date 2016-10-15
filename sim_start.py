import matplotlib.pylab as py
import math
import numpy as np


class scene:
	def __init__(self, traits):
		(eps, lam, B) = traits
		self.eps = eps*1.0
		self.lam = lam*1.0
		self.B = B*1.0

		self.dict_oneShotDyn = {}
		self.dict_oneShotRev = {}
		self.dict_profit = {}


	def findOptPrice(self):
		l = math.exp(-1)
		u = math.exp(-0.5)
		
		goal = self.lam * self.B

		self.rStar = binarySearch(1, l, u, goal, lambda x: 2*x*math.log(x) + x)
		
		self.pStar = self.rStar/(2*self.lam) + self.B/2

	
		#critical point for pStar	
		l = 0.0
		u = math.exp(-1.0)
		goal = -self.lam*(self.pStar - self.B)
		self.rCrit = binarySearch(-1, l, u, goal, lambda x: x*math.log(x)) 
					

	def oneShotDynamic(self, T, p0, gamma):
		
		dynamic = np.empty(T)
		dynamic[0] = self.eps

		dynamic[1] = growth(dynamic[0], p0, self.B, self.lam)
		
		for i in range(2,T):
			dynamic[i] = growth(dynamic[i-1], self.pStar, self.B, self.lam)

		self.dict_oneShotDyn[(T, p0)] = dynamic 
		
		profit = [x*self.pStar for x in dynamic]
		profit[0] = 0.0  #fix for error, because we only start counting from T=1
		profit[1] = p0*dynamic[1] 
		
		self.dict_oneShotRev[(T, p0)] = profit
		
		self.dict_profit[(T, p0)] = discounted_profit(gamma, profit)
		


	#go from eps to rCrit
	def find_pLow(self):
		self.pLow = (1/self.lam)*(-self.eps*math.log(self.rCrit) + self.lam*self.B)
		
#
#d = 1 ->increaseing function
#d =-1 ->decreasing function
def binarySearch(d, l, u, goal,f):

	if d ==1:
		while u-l>0.00001:
			m=(l+u)/2.0
			val = f(m)

			if val > goal:
				u = m

			else:
				l=m

	else:	
		while u-l>0.00001:
			m=(l+u)/2.0
			val = f(m)

			if val < goal:
				u = m

			else:
				l=m


	return (l+u)/2.0




def growth(r, p, B, lam):

	if r >0.0000001:
		try:
			r_next = math.exp(-lam*1.0*(p-B)/r)

		except:
			print r
			print p
			print B
	else:
		if p-B<0:
			r_next = 1.0
		else:
			r_next = 0.0
	return min(1.0, r_next)



def discounted_profit(gamma, profit):
	t=len(profit)
	total = 0.0
	for i in range(1,t):
		total = total + profit[i] * gamma**(i-1)

	return total


#======================================================

#trait = eps, lambda, B

trait = (0.0001, 1.0/20,-5.0)
scene1 = scene(trait)


scene1.findOptPrice()
scene1.find_pLow()
print "scene1 pStar"+str(scene1.pStar)
print "scene1 rStar" + str(scene1.rStar)
print "scene1 rCrit" + str(scene1.rCrit)
print "scene1 pLow" + str(scene1.pLow)

#================================================

for T in [15, 30]:

	gamma = 0.9

	step = (scene1.pLow-scene1.B)/15
	xAxis = [i for i in range(0,T)]

	for i in range(0,15):
		p0 = scene1.pLow - i*step
		scene1.oneShotDynamic(T,p0,gamma)
		#print p0
		#print scene1.dict_profit[(T,p0)]
		#print scene1.dict_oneShotDyn[(T, p0)]
	
		name = str(p0*10000)[1:6]

		fig = py.figure()
		py.xlabel("time")
		py.ylabel("participation")
		py.axis([0,T,0,1])
		py.plot(xAxis, scene1.dict_oneShotDyn[(T,p0)])
		fig.savefig("Scene1_Dynamic_p0_"+name+"_T_"+str(T)+".png")
		py.close(fig)

		fig = py.figure()
		py.xlabel("time")
		py.ylabel("revenue")
		py.axis([0,T,-5.5,3])
		py.plot(xAxis, scene1.dict_oneShotRev[(T,p0)])
		fig.savefig("Scene1_Rev_p0_"+name+"_T_"+str(T)+".png")
		py.close(fig)

	x_p0 = [x[1] for x in scene1.dict_profit]
	x_p0.sort()
	y_profit = [scene1.dict_profit[(T,i)] for i in x_p0]

	fig = py.figure()
	py.xlabel("initialPrice")
	py.ylabel("discounted profit")
	py.axis([-5.00, -4.997,1,7])
	py.plot(x_p0,y_profit)
	fig.savefig("Scene1_changeDisPro"+str(gamma)+"_T_"+str(T)+".png")
	py.close(fig)







#======================================================

#trait = eps, lambda, B

trait = (0.0001, 1.0, -0.2)
scene2 = scene(trait)


scene2.findOptPrice()
scene2.find_pLow()
print "scene2 pLow" + str(scene2.pLow)
print "scene2 rStar" + str(scene2.rStar)
print "scene2 rCrit" + str(scene2.rCrit)
print "scene2 pStar" + str(scene2.pStar)
#================================================

for T in [15, 30]:

	gamma = 0.9

	step = (scene2.pLow - scene2.B)/15
	xAxis = [i for i in range(0,T)]

	for i in range(0,15):
		p0 = scene2.pLow - i*step
		scene2.oneShotDynamic(T,p0,gamma)
		#print p0
		#print scene1.dict_profit[(T,p0)]
		#print scene1.dict_oneShotDyn[(T, p0)]
	
		name = str(p0*10000)[1:6]

		fig = py.figure()
		py.xlabel("time")
		py.ylabel("participation")
		py.axis([0,T,0,1])
		py.plot(xAxis, scene2.dict_oneShotDyn[(T,p0)])
		fig.savefig("Scene2_Dynamic_p0_"+name+"_T_"+str(T)+".png")
		py.close(fig)

		fig = py.figure()
		py.xlabel("time")
		py.ylabel("revenue")
		py.axis([0,T,-0.2,0.2])
		py.plot(xAxis, scene2.dict_oneShotRev[(T,p0)])
		fig.savefig("Scene2_Rev_p0_"+name+"_T_"+str(T)+".png")
		py.close(fig)

	x_p0 = [x[1] for x in scene2.dict_profit]
	x_p0.sort()
	y_profit = [scene2.dict_profit[(T,i)] for i in x_p0]

	fig = py.figure()
	py.xlabel("initialPrice")
	py.ylabel("discounted profit")
	py.axis([scene2.B, scene2.pLow,0,1])
	py.plot(x_p0,y_profit)
	fig.savefig("Scene2_changeDisPro"+str(gamma)+"_T_"+str(T)+".png")
	py.close(fig)





#======================================================

#trait = eps, lambda, B

trait = (0.0001, 1.0/10.0,-3.0)
scene3 = scene(trait)


scene3.findOptPrice()
scene3.find_pLow()
print "scene3 pStar"+str(scene3.pStar)
print "scene3 rStar" + str(scene3.rStar)
print "scene3 rCrit" + str(scene3.rCrit)
print "scene3 pLow" + str(scene3.pLow)

#================================================

for T in [15, 30]:

	gamma = 0.9

	step = (scene3.pLow-scene3.B)/15
	xAxis = [i for i in range(0,T)]

	for i in range(0,15):
		p0 = scene3.pLow - i*step
		scene3.oneShotDynamic(T,p0,gamma)
		#print p0
		#print scene1.dict_profit[(T,p0)]
		#print scene1.dict_oneShotDyn[(T, p0)]
	
		name = str(p0*10000)[1:6]

		fig = py.figure()
		py.xlabel("time")
		py.ylabel("participation")
		py.axis([0,T,0,1])
		py.plot(xAxis, scene3.dict_oneShotDyn[(T,p0)])
		fig.savefig("Scene3_Dynamic_p0_"+name+"_T_"+str(T)+".png")
		py.close(fig)

		fig = py.figure()
		py.xlabel("time")
		py.ylabel("revenue")
		py.axis([0,T,-2,0.5])
		py.plot(xAxis, scene3.dict_oneShotRev[(T,p0)])
		fig.savefig("Scene3_Rev_p0_"+name+"_T_"+str(T)+".png")
		py.close(fig)

	x_p0 = [x[1] for x in scene3.dict_profit]
	x_p0.sort()
	y_profit = [scene3.dict_profit[(T,i)] for i in x_p0]

	fig = py.figure()
	py.xlabel("initialPrice")
	py.ylabel("discounted profit")
	py.axis([scene3.B, scene3.pLow ,-1,3])
	py.plot(x_p0,y_profit)
	fig.savefig("Scene3_changeDisPro"+str(gamma)+"_T_"+str(T)+".png")
	py.close(fig)








#======================================================

#trait = eps, lambda, B

trait = (0.0001, 1.0/10,-1.0)
scene4 = scene(trait)


scene4.findOptPrice()
scene4.find_pLow()
print "scene4 pStar"+str(scene4.pStar)
print "scene4 rStar" + str(scene4.rStar)
print "scene4 rCrit" + str(scene4.rCrit)
print "scene4 pLow" + str(scene4.pLow)

#================================================

for T in [15, 30]:

	gamma = 0.9

	step = (scene4.pLow-scene4.B)/15
	xAxis = [i for i in range(0,T)]

	for i in range(0,15):
		p0 = scene4.pLow - i*step
		scene4.oneShotDynamic(T,p0,gamma)
		#print p0
		#print scene1.dict_profit[(T,p0)]
		#print scene1.dict_oneShotDyn[(T, p0)]
	
		name = str(p0*10000)[1:6]

		fig = py.figure()
		py.xlabel("time")
		py.ylabel("participation")
		py.axis([0,T,0,1])
		py.plot(xAxis, scene4.dict_oneShotDyn[(T,p0)])
		fig.savefig("Scene4_Dynamic_p0_"+name+"_T_"+str(T)+".png")
		py.close(fig)

		fig = py.figure()
		py.xlabel("time")
		py.ylabel("revenue")
		py.axis([0,T,-1.0,2])
		py.plot(xAxis, scene4.dict_oneShotRev[(T,p0)])
		fig.savefig("Scene4_Rev_p0_"+name+"_T_"+str(T)+".png")
		py.close(fig)

	x_p0 = [x[1] for x in scene4.dict_profit]
	x_p0.sort()
	y_profit = [scene4.dict_profit[(T,i)] for i in x_p0]

	fig = py.figure()
	py.xlabel("initialPrice")
	py.ylabel("discounted profit")
	py.axis([scene4.B, scene4.pLow,2,12])
	py.plot(x_p0,y_profit)
	fig.savefig("Scene4_changeDisPro"+str(gamma)+"_T_"+str(T)+".png")
	py.close(fig)

	print "gamma 0.9"
	print y_profit


#scene 4 with a lower gamma
for T in [15, 30]:

	gamma = 0.8


	x_p0 = [x[1] for x in filter(lambda x: x[0]==T, scene4.dict_oneShotRev)]
	x_p0.sort()
	
	y_profit = [discounted_profit(gamma, scene4.dict_oneShotRev[(T, i)]) for i in x_p0]

	fig = py.figure()
	py.xlabel("initialPrice")
	py.ylabel("discounted profit")
	py.axis([scene4.B, scene4.pLow,0,5])
	py.plot(x_p0,y_profit)
	fig.savefig("Scene4_changeDisPro"+str(gamma)+"_T_"+str(T)+".png")
	py.close(fig)

	print "gamma 0.8"
	print y_profit


#scene 4 with a lower gamma
for T in [15, 30]:

	gamma = 0.7


	x_p0 = [x[1] for x in filter(lambda x: x[0]==T, scene4.dict_oneShotRev)]
	x_p0.sort()
	
	y_profit = [discounted_profit(gamma, scene4.dict_oneShotRev[(T, i)]) for i in x_p0]

	fig = py.figure()
	py.xlabel("initialPrice")
	py.ylabel("discounted profit")
	py.axis([scene4.B, scene4.pLow,-1,5])
	py.plot(x_p0,y_profit)
	fig.savefig("Scene4_changeDisPro"+str(gamma)+"_T_"+str(T)+".png")
	py.close(fig)


	print "gamma 0.7"
	print y_profit
#def discounted_profit(gamma, profit):
#	t=len(profit)
#	total = 0.0
#	for i in range(1,t):
#		total = total + profit[i] * gamma**(i-1)
#
#	return total
