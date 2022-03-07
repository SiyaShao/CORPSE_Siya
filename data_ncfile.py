import netCDF4 as nc
import numpy as np
import pandas as pd

def data_ncfile(placename):

    filepath = 'D:/Postdoc/ELM sims_for_Siya/sims_for_Siya/'
    file = filepath + placename + '.nc'
    dataset = nc.Dataset(file)
    # print(dataset.variables.keys())

    converter = 3600*24  # Convert secondly rate to daily
    if placename == 'US-Ho1':
        LeafCN = 40.0  # needleleaf evergreen boreal trees
    else:
        LeafCN = 25.0  # temperate deciduous trees
    RootCN = 42.0
    Leaf_litterCN = LeafCN * 2.0
    Root_litterCN = RootCN
    PlantNdemand = dataset.variables['PLANT_NDEMAND'][:]*converter
    Nrootuptake = dataset.variables['SMINN_TO_PLANT'][:]*converter
    NPP = dataset.variables['NPP'][:]*converter
    LeafC = dataset.variables['LEAFC'][:]
    LeafN = LeafC / LeafCN
    LitterC = dataset.variables['LITFALL'][:]*converter
    Leaf_litterC = dataset.variables['LEAFC_TO_LITTER'][:]*converter
    Root_litterC = LitterC - Leaf_litterC
    RootC = (LitterC - Leaf_litterC) / Leaf_litterC * LeafC
    RootN = (RootC / RootCN)
    LitterCN = LitterC / (Leaf_litterC / Leaf_litterCN + Root_litterC / Root_litterCN)
    LitterN = (LitterC / LitterCN)
    LitterC = LitterC
    LitterN = LitterN
    LeafN = LeafN
    SoilT = dataset.variables['TSOI_10CM'][:] - 273.15
    SoilM = dataset.variables['H2OSOI'][:, 3, 0]

    result_df = pd.DataFrame({'NPP': NPP.ravel(), 'LeafN': LeafN.ravel(), 'RootN': RootN.ravel(), 'PlantNdemand': PlantNdemand.ravel(),
                              'Nrootuptake': Nrootuptake.ravel(), 'LitterC': LitterC.ravel(), 'LitterN': LitterN.ravel(),
                              'SoilT': SoilT.ravel(), 'SoilM': SoilM.ravel()})
    return result_df

if __name__ == '__main__':
    # Test simulation
    import matplotlib.pyplot as plt
    results = data_ncfile('US-Ho1')
    time = np.linspace(0, 1, len(results['NPP']))
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