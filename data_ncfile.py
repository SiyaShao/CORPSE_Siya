import pandas as pd
import numpy as np

def data_ncfile(name):
    #name = 'CW-CS01'
    filename = 'D:/Postdoc/CORPSE_Siya/ELMoutputs/' + name + '.txt'
    file = open(filename, "r")
    datastring = file.read()
    datalist = datastring.split("\n")

    timesteps = 120
    NPP = np.zeros(timesteps)
    LeafN = np.zeros(timesteps)
    RootN = np.zeros(timesteps)
    PlantNdemand = np.zeros(timesteps)
    Nrootuptake = np.zeros(timesteps)
    LitterC = np.zeros(timesteps)
    LitterN = np.zeros(timesteps)
    SoilT = np.zeros(timesteps)
    SoilM = np.zeros(timesteps)
    for i in range(0, timesteps, 1):
        data = datalist[i].split(" ")
        # print(data,i)
        NPP[i] = data[0]
        LeafN[i] = data[1]
        RootN[i] = data[2]
        PlantNdemand[i] = data[3]
        Nrootuptake[i] = data[4]
        LitterC[i] = data[5]
        LitterN[i] = data[6]
        SoilT[i] = data[7]
        SoilM[i] = data[8]
    d = {'NPP': NPP, 'LeafN': LeafN, 'RootN': RootN, 'PlantNdemand': PlantNdemand, 'Nrootuptake': Nrootuptake,
         'LitterC': LitterC, 'LitterN': LitterN, 'SoilT': SoilT,  'SoilM': SoilM}
    result_df = pd.DataFrame(data=d)

    return result_df

if __name__ == '__main__':
    # Test simulation
    import matplotlib.pyplot as plt
    results = data_ncfile('US-Ho1')
    timesteps = 120
    time = np.linspace(0, 1, timesteps)
    plt.subplot(2,3,1)
    plt.plot(time,results['NPP'])
    plt.xlabel("Time (month)")
    plt.ylabel("C fluxes (gC/m2/month)")
    plt.legend(["NPP"])
    plt.subplot(2,3,2)
    plt.plot(time,results['LeafN'])
    plt.plot(time,results['RootN'])
    plt.xlabel("Time (month)")
    plt.ylabel("N stocks (gN/m2)")
    plt.legend(["Leaf N","Root N"])
    plt.subplot(2,3,3)
    plt.plot(time,results['PlantNdemand'])
    plt.plot(time,results['Nrootuptake'])
    plt.xlabel("Time (month)")
    plt.ylabel("N fluxes (gN/m2/month)")
    plt.legend(["Plant Ndemand","Plant Nrootuptake"])
    ax1 = plt.subplot(2,3,4)
    ax2 = ax1.twinx()
    ax1.plot(time,results['LitterC'],label = 'Litter C')
    ax2.plot(time,results['LitterN'],label = 'Litter N')
    ax1.plot(np.nan, '-r', label = 'Litter N')  # Make an agent in ax
    ax1.legend(loc=0)
    ax1.set_xlabel("Time (month)")
    ax1.set_ylabel("Litter C fluxes (gC/m2/month)")
    ax2.set_ylabel("Litter N fluxes (gN/m2/month)")
    plt.subplot(2,3,5)
    plt.plot(time,results['SoilT'])
    plt.xlabel("Time (month)")
    plt.ylabel("T (°C)")
    plt.legend(["Soil Temperature"])
    plt.subplot(2,3,6)
    plt.plot(time,results['SoilM'])
    plt.xlabel("Time (month)")
    plt.ylabel("Moisture (mm3/mm3)")
    plt.legend(["Soil Moisture"])
    plt.show()