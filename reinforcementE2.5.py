import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

####################################################
# Reward Distribution
####################################################
RD = []  # Reward Distribution
k = 10  # nr of choices:

# calculate random rewards:
for x in range(0, k-1):
    RD.append(np.random.normal(0.0, 1.0))



# This simulates a pick from the Reward distribution


def pick_and_reward(n, RD=RD):
    return np.random.normal(RD[n], 1)

#original
"""
def simulate(rd, nr_picks, epsilon, sigma, percent_correct_pick, Qn):
    k = len(rd)
    # del percent_correct_pick[:]
    # del Qn[:]
    Qa = np.zeros(k)
    Na = np.zeros(k)
    Qn.extend(np.zeros(nr_picks))
    percent_correct_pick.extend(np.zeros(nr_picks))
    correct_pick = np.argmax(rd, 0)
    for step in range(1, nr_picks):
        if (np.random.random() <= epsilon):
            choice = np.random.randint(0, k)
        else:
            # get the highest reward:
            choice = np.argmax(Qa, 0)
        reward = np.random.normal(rd[choice], sigma)
        Na[choice]=Na[choice]+1
        Qa[choice] = Qa[choice]+(1/Na[choice]*(reward-Qa[choice]))
        Qn[step] = Qn[step-1] + 1/step * (reward - Qn[step-1])
        if (correct_pick == choice):
            cor = 1
        else:
            cor = 0
        percent_correct_pick[step] = percent_correct_pick[step -1] + 1/step * (cor - percent_correct_pick[step-1])
"""

#fixed alpha:
alpha=1/20
def simulate(rd, nr_picks, epsilon, sigma, percent_correct_pick, Qn):
    k = len(rd)
    # del percent_correct_pick[:]
    # del Qn[:]
    Qa = np.zeros(k)
    Na = np.zeros(k)
    Qn.extend(np.zeros(nr_picks))
    percent_correct_pick.extend(np.zeros(nr_picks))
    correct_pick = np.argmax(rd, 0)
    for step in range(1, nr_picks):
        if (np.random.random() <= epsilon):
            choice = np.random.randint(0, k)
        else:
            # get the highest reward:
            choice = np.argmax(Qa, 0)
        reward = np.random.normal(rd[choice], sigma)
        Na[choice]=Na[choice]+1
        Qa[choice] = Qa[choice]+(alpha*(reward-Qa[choice]))
        Qn[step] = Qn[step-1] + 1/step * (reward - Qn[step-1])
        if (correct_pick == choice):
            cor = 1
        else:
            cor = 0
        percent_correct_pick[step] = percent_correct_pick[step -1] + 1/step * (cor - percent_correct_pick[step-1])



sigma = 1
pcp_medium = []
qn_medium = []
k = 10
N_RUNS = 15
N_STEPS = 10000

pcp_medium = np.zeros(N_STEPS)
qn_medium = np.zeros(N_STEPS)


if 1:
    for run_i in tqdm(range(1, N_RUNS)):
        pcp = []
        qn = []
        RD = []
        # recalculate RD:
        RD.extend(np.random.normal(0.0, 1.0, k))
        simulate(RD, N_STEPS, 0.1, sigma, pcp, qn)

        pcp_medium = pcp_medium + 1/run_i * (pcp-pcp_medium)
        qn_medium = qn_medium + 1/run_i * (qn-qn_medium)
    fig, axs = plt.subplots(2)
    fig.suptitle('k-armed Bandit')
    axs[0].plot(range(0, N_STEPS), qn_medium, 'r')
    axs[1].plot(range(0, N_STEPS), pcp_medium, 'r')

    
    pcp_medium = np.zeros(N_STEPS)
    qn_medium = np.zeros(N_STEPS)
    for run_i in tqdm(range(1, N_RUNS)):
        pcp = []
        qn = []
        RD = []
        # recalculate RD:
        RD.extend(np.random.normal(0.0, 1.0, k))
        simulate(RD, N_STEPS, 0.01, sigma, pcp, qn)

        pcp_medium = pcp_medium + 1/run_i * (pcp-pcp_medium)
        qn_medium = qn_medium + 1/run_i * (qn-qn_medium)    
    axs[0].plot(range(0, N_STEPS), qn_medium, 'b')
    axs[1].plot(range(0, N_STEPS), pcp_medium, 'b')

    pcp_medium = np.zeros(N_STEPS)
    qn_medium = np.zeros(N_STEPS)
    for run_i in tqdm(range(1, N_RUNS)):
        pcp = []
        qn = []
        RD = []
        # recalculate RD:
        RD.extend(np.random.normal(0.0, 1.0, k))
        simulate(RD, N_STEPS, 0.3, sigma, pcp, qn)

        pcp_medium = pcp_medium + 1/run_i * (pcp-pcp_medium)
        qn_medium = qn_medium + 1/run_i * (qn-qn_medium)    
    axs[0].plot(range(0, N_STEPS), qn_medium, 'c')
    axs[1].plot(range(0, N_STEPS), pcp_medium, 'c')


    plt.show()

####################################################
#
#
####################################################
# Picking simulation
####################################################
"""nr_picks = 100000 #number of picks

Qa = np.zeros(k)  # Reward distribution

Qn = np.zeros(nr_picks) #average reward

epsilon = 0.01
for step in range(1,nr_picks):
    if (np.random.random() <= epsilon):
        choice = np.random.randint(0,k-1)
    else:
        # get the highest reward:
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
