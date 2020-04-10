import numpy as np
import matplotlib.pyplot as plt 

####################################################
#Reward Distribution
####################################################
RD = [] # Reward Distribution
k=10 #nr of choices:

#calculate random rewards:
for x in range(0,k-1):
    RD.append(np.random.normal(0.0,1.0))

#print rewards
print ("Rewards:")
print (RD)

#This simulates a pick from the Reward distribution
def pick_and_reward(n):
    return np.random.normal(RD[n],1)
####################################################
#
#
####################################################
#Picking simulation
####################################################
nr_picks = 100000 #number of picks

Qa = np.zeros(k)  # Reward distribution

Qn = np.zeros(nr_picks) #average reward

epsilon = 0.01
for step in range(1,nr_picks):
    if (np.random.random() <= epsilon):
        choice = np.random.randint(0,k-1)
    else:
        #get the highest reward:
        choice = np.argmax(Qa, 0)
    reward = pick_and_reward(choice)
    Qa[choice]=Qa[choice]+(1/step*(reward-Qa[choice]))
    Qn[step]= Qn[step-1] + 1/step * (reward - Qn[step-1])
    


print(Qa)

print(RD)

print ( "choice in the end:")
print (np.argmax(Qa,0))

plt.plot(range(0,nr_picks),Qn)
plt.show()
    


"""
xA = []
xA = []
yA = []
for x in range (1,100):
    xA.append(x)
    
    a=np.random.normal(1,3)
    yA.append(a)
    print(a)

plt.plot(xA,yA)

plt.show()
"""
