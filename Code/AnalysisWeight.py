import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def AnalyzeDitriWeight(Weig):
    r, c = Weig.shape
    print(r,c)

    q=np.zeros((1,c))
    P = np.zeros((1, c))
    #print(q[0])
    #exit()
    for i in range(r):
        for j in range(c):
            #print("=======",j,"=++++++++++++++++",i)
            q[0][j]=q[0][j]+abs(Weig[i][j])

    for j in range(c):
        if q[0][j]<=46.523:
            P[0][j]=-7
        elif q[0][j]<=50:
            P[0][j]=2
        elif q[0][j]<=54:
            P[0][j]=5
        else:
            P[0][j]=7

    '''
    fig, ax = plt.subplots()
    ax.plot(q[0])
    plt.show()'''
    #exit()
    return  P[0]
