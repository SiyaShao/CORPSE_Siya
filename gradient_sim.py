import pandas, numpy

# Code for integrating the model using python ODE solver
import CORPSE_integrate

# Initial values for all pools in the model
# The {} brackets create a "dictionary" where everything is indexed by a name rather than a number
# Pools starting with "u" are unprotected and pools starting with "p" are protected
# These are assumed to be ugC/g mineral as in the incubation. But the units are not that important as long as they are
# consistent throughout

SOM_init = {'uFastC': 0.1,
            'uSlowC': 10.0,
            'uNecroC': 0.1,
            'pFastC': 0.1,
            'pSlowC': 0.1,
            'pNecroC': 10.0,
            'SAPC': 0.01,
            'uFastN': 0.1e-1,
            'uSlowN': 20.0e-1,
            'uNecroN': 0.1e-1,
            'pFastN': 10.0e-1,
            'pSlowN': 0.1e-1,
            'pNecroN': 10.0e-1,
            # 'SAPN':0.01/8.0,
            'inorganicN': 0.1,
            'CO2': 0.0,
            'ECMC': 0.025,
            'AMC': 0.025,
            'Int_ECMC': 0.0,
            'Int_AMC': 0.0,
            'Int_N': 0.0, # Not splitting intermediate N pool between ECM and AM for now
            'NfromNecro': 0.0,
            'NfromSOM': 0.0,
            'Nlimit': 0.0}

# Set model parameters
# Note that carbon types have names, in contrast to previous version
params = {
    'vmaxref': {'Fast': 9.0, 'Slow': .5, 'Necro': 4.5},  # Relative maximum enzymatic decomp rates (year-1)
    'Ea': {'Fast': 5e3, 'Slow': 30e3, 'Necro': 3e3},
    # Activation energy (Controls temperature sensitivity via Arrhenius relationship)
    'kC': {'Fast': 0.01, 'Slow': 0.01, 'Necro': 0.01},  # Michaelis-Menton parameter
    'gas_diffusion_exp': 0.6,  # Determines suppression of decomp at high soil moisture
    'substrate_diffusion_exp': 1.5,  # Dimensionless exponent. Determines suppression of decomp at low soil moisture
    'minMicrobeC': 1e-3,  # Minimum microbial biomass (fraction of total C)
    'Tmic': {'SAP': 0.25, 'ECM': 1.0, 'AM': 1.0},  # Microbial lifetime (years)
    'et': {'SAP': 0.6, 'ECM': 0.6, 'AM': 0.6},  # Fraction of turnover not converted to CO2 (dimensionless)
    'eup': {'Fast': 0.6, 'Slow': 0.05, 'Necro': 0.6},  # Carbon uptake efficiency (dimensionless fraction)
    'tProtected': 75.0,  # Protected C turnoveir time (years)
    'protection_rate': {'Fast': 0.1, 'Slow': 0.0001, 'Necro': 1.5},  # Protected carbon formation rate (year-1)
    'new_resp_units': True,
    'frac_N_turnover_min': 0.2,
    'frac_turnover_slow': {'SAP': 0.2, 'ECM': 0.2, 'AM': 0.2},
    'nup': {'Fast': 0.9, 'Slow': 0.6, 'Necro': 0.9},
    'CN_microbe': {'SAP':8.0,'ECM':10.0,'AM':10.0},
    'max_immobilization_rate': 3.65,
    'substrate_diffusion_exp': 1.5,
    'new_resp_units': True,
    'eup_myc': {'ECM':0.5,'AM':0.5},
    'max_mining_rate': {'Fast': 9.0, 'Slow': 1.0, 'Necro': 9.0},
    # assumed to be more efficient with Slow and Necro pools, and less efficient with fast pools, compared to SAPs
    'kc_mining': 0.015, # g microbial biomass C/g substrate C, Sulman et al., (2019)
    'max_scavenging_rate': {'SAP':0.16,'ECM':0.0,'AM':0.2},
    # kgN/m2/year, AM value is from Sulman et al., (2019), assumed 80% AM capacity for SAP N immobilization, 0 for ECM
    'kc_scavenging': {'SAP':0.3,'ECM':0.3,'AM':0.3},
    # kgC/m3, AM value is from Sulman et al., (2019), assumed to be the same for other microbes
    'kc_scavenging_IN': {'SAP':0.001,'ECM':0.001,'AM':0.001},
    # kgN/m3, AM value is from Sulman et al., (2019), assumed to be the same for other microbes
    'Ea_inorgN': 37e3,  # (kJ/mol) Activation energy for immobilization of inorganic N from Sulman et al., (2019)
    'depth': 0.1, # 10cm, assumed for now.
    'iN_loss_rate': 10.0, # Loss rate from inorganic N pool (year-1). >1 since it takes much less than a year for it to be removed
    'N_deposition': 0.001,  # 1.0gN/m2/yr
    # Loss rate from inorganic N pool (year-1). >1 since it takes much less than a year for it to be removed
    'Ohorizon_transfer_rates': {'uFastC': 0.1, 'uSlowC': 0.1, 'uNecroC': 0.1, 'uFastN': 0.1, 'uSlowN': 0.1,
                                'uNecroN': 0.1}
}
SOM_init['SAPN'] = SOM_init['SAPC'] / params['CN_microbe']['SAP']
SOM_init['ECMN'] = SOM_init['ECMC'] / params['CN_microbe']['ECM']
SOM_init['AMN'] = SOM_init['AMC'] / params['CN_microbe']['AM']

# ECM gradient plots
nplots = 20
nclays = 2
nclimates = 3
# Environmental conditions
# Gradient of mycorrhizal association
ECM_pct = numpy.linspace(0, 100, nplots)  # Percent ECM basal area
MAT = numpy.linspace(5, 20, nclimates)  # degrees C
clay = numpy.linspace(10, 70, nclays)  # percent clay

fastfrac_AM = 0.4
fastfrac_ECM = 0.1
fastfrac_site = fastfrac_ECM * ECM_pct / 100 + fastfrac_AM * (1 - ECM_pct / 100)
litter_CN_AM = 30
litter_CN_ECM = 50
litter_CN_site = litter_CN_ECM * ECM_pct / 100 + litter_CN_AM * (1 - ECM_pct / 100)
total_inputs = 0.5  # kgC/m2

fastfrac_site[:] = 0.1
litter_CN_site = 50

myc_ratio_NPP = 0.2  # Ratio of C transferred to mycorrhizal fungi to NPP
myc_ratio_litter = myc_ratio_NPP / (
            1 - myc_ratio_NPP)  # Ratio of C transferred to mycorrhizal fungi to litter production

inputs = {'uFastC': total_inputs * fastfrac_site,
          'uSlowC': total_inputs * (1 - fastfrac_site),
          'uFastN': total_inputs * fastfrac_site / litter_CN_site,
          'uSlowN': total_inputs * (1 - fastfrac_site) / litter_CN_site,
          'Int_ECMC': total_inputs * myc_ratio_litter * ECM_pct / 100,
          'Int_AMC': total_inputs * myc_ratio_litter * (1 - ECM_pct / 100)}

Ctransfer = {'ECM':0.0,'AM':0.0} # Initialize C tranfer from plants to symbiont fungi

theta = 0.5  # fraction of saturation

spinuptimes = numpy.arange(0, 2500,
                     10)  # 10-year time steps for model spinup as spinning up on finer timestep would take too long
timestep = 0.25  # Quarterly
finaltimes = numpy.arange(0, 100 + timestep, timestep)  # Time steps to evaluat, running on quatily timestep
plottimes = finaltimes
timesteps = len(finaltimes)  # According to what is set for times in the above lines (numpy.arrange)
# The ODE solver uses an adaptive timestep but will return these time points


# Run the simulations
protC = numpy.zeros((nplots, nclays, nclimates))
protN = numpy.zeros((nplots, nclays, nclimates))
unprotC = numpy.zeros((nplots, nclays, nclimates))
unprotN = numpy.zeros((nplots, nclays, nclimates))
inorgN = numpy.zeros((nplots, nclays, nclimates))
sapC = numpy.zeros((nplots, nclays, nclimates))
ecmC = numpy.zeros((nplots, nclays, nclimates))
amC = numpy.zeros((nplots, nclays, nclimates))
microbC = numpy.zeros((nplots, nclays, nclimates))

SAPC = numpy.zeros((timesteps, nplots, nclays, nclimates))
ECMC = numpy.zeros((timesteps, nplots, nclays, nclimates))
AMC = numpy.zeros((timesteps, nplots, nclays, nclimates))  # Document the mycorrhizal changes with time
UnprotectedC = numpy.zeros((timesteps, nplots, nclays, nclimates))
UnFastC = numpy.zeros((timesteps, nplots, nclays, nclimates))
UnSlowC = numpy.zeros((timesteps, nplots, nclays, nclimates))
UnNecroC = numpy.zeros((timesteps, nplots, nclays, nclimates))
ProtectedC = numpy.zeros((timesteps, nplots, nclays, nclimates))
PFastC = numpy.zeros((timesteps, nplots, nclays, nclimates))
PSlowC = numpy.zeros((timesteps, nplots, nclays, nclimates))
PNecroC = numpy.zeros((timesteps, nplots, nclays, nclimates))
InorgN = numpy.zeros((timesteps, nplots, nclays, nclimates))

def experiment(plotnum,claynum,climnum):
    from CORPSE_deriv import sumCtypes
    print(
        'Sim {simnum:d} of {totsims:d}. %ECM = {ecmpct:1.1f}, %clay = {claypct:1.1f}, MAT = {mat:1.1f}'.format(
            simnum=6*plotnum+3*claynum+climnum+1, totsims=nplots*nclays*nclimates, ecmpct=ECM_pct[plotnum], claypct=clay[claynum],
            mat=MAT[climnum]))

    Ndemand = total_inputs/litter_CN_site
    # Calculate plant Ndemand from N in litter production
    # Assuming plant N_litter balances plant N_uptake and plant not relying on roots
    # This will not be needed once the model is coupled with a plant model

    Ctransfer['ECM'] = total_inputs * myc_ratio_litter * ECM_pct[plotnum] / 100
    Ctransfer['AM'] = total_inputs * myc_ratio_litter * (1-ECM_pct[plotnum] / 100)

    result = CORPSE_integrate.run_CORPSE_ODE(T=MAT[climnum], theta=theta, Ndemand=Ndemand,
                                             inputs=dict([(k, inputs[k][plotnum]) for k in inputs]),
                                             clay=clay[claynum], initvals=SOM_init, params=params,
                                             times=spinuptimes, runtype='Spinup')
    result = CORPSE_integrate.run_CORPSE_ODE(T=MAT[climnum], theta=theta, Ndemand=Ndemand,
                                             inputs=dict([(k, inputs[k][plotnum]) for k in inputs]),
                                             clay=clay[claynum], initvals=result.iloc[-1], params=params,
                                             times=finaltimes, runtype='Final')
    SAPCarray = result['SAPC']
    ECMCarray = result['ECMC']
    AMCarray = result['AMC']
    UnFastCCarray = result['uFastC']
    UnSlowCCarray = result['uSlowC']
    UnNecroCCarray = result['uNecroC']
    UnprotectedCarray = sumCtypes(result.iloc[:], 'u', 'C')
    PFastCCarray = result['pFastC']
    PSlowCCarray = result['pSlowC']
    PNecroCCarray = result['pNecroC']
    ProtectedCarray = sumCtypes(result.iloc[:], 'p', 'C')
    InorgNarray = result['inorganicN']
    IntNarray = result['Int_N']
    TotNarray = sumCtypes(result.iloc[:], 'u', 'N') + sumCtypes(result.iloc[:], 'p', 'N')
    NfromNecroarray = result['NfromNecro']
    NfromSOMarray = result['NfromSOM']
    Nlimitarray = result['Nlimit']
    for timenum in range(timesteps):
        filename = str(6 * plotnum + 3 * claynum + climnum + 1) + "_Quarterly_data.txt"
        if timenum == 0:
            f = open(filename, "w")
        else:
            f = open(filename, "a")
        f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(InorgNarray[timenum],
                                                                                          SAPCarray[timenum],
                                                                            UnprotectedCarray[timenum], ECMCarray[timenum],
                                                                            AMCarray[timenum],
                                                                            TotNarray[timenum], NfromNecroarray[timenum],
                                                                            NfromSOMarray[timenum],Nlimitarray[timenum]))
    f.close()

from joblib import Parallel, delayed
output = Parallel(n_jobs=10)(
    delayed(experiment)(plotnum, claynum, climnum) for plotnum in range(nplots) for claynum in range(nclays) for
    climnum in range(nclimates))

# Plot the results
import matplotlib.pyplot as plt

norm = plt.Normalize(5, 20)
normECM = plt.Normalize(0, 100)
cmap = plt.get_cmap('rainbow')
cmapECM = plt.get_cmap('summer')
markers = ['o', 's']
SMALL_SIZE = 8

plot_InorgN = numpy.zeros([nplots, nclays, nclimates])
plot_SAPC = numpy.zeros([nplots, nclays, nclimates])
plot_UnpC = numpy.zeros([nplots, nclays, nclimates])
plot_ECMC = numpy.zeros([nplots, nclays, nclimates])
plot_AMC = numpy.zeros([nplots, nclays, nclimates])
plot_TotN = numpy.zeros([nplots, nclays, nclimates])
plot_NfromNecro = numpy.zeros([nplots, nclays, nclimates])
plot_NfromSOM = numpy.zeros([nplots, nclays, nclimates])
plot_Nsource = numpy.zeros([nplots, nclays, nclimates])
plot_Nlimit = numpy.zeros([nplots, nclays, nclimates])

plot_InorgN_all = numpy.zeros([timesteps, nplots, nclays, nclimates])
plot_SAPC_all = numpy.zeros([timesteps, nplots, nclays, nclimates])
plot_UnpC_all = numpy.zeros([timesteps, nplots, nclays, nclimates])
plot_ECMC_all = numpy.zeros([timesteps, nplots, nclays, nclimates])
plot_AMC_all = numpy.zeros([timesteps, nplots, nclays, nclimates])
plot_TotN_all = numpy.zeros([timesteps, nplots, nclays, nclimates])
plot_NfromNecro_all = numpy.zeros([timesteps, nplots, nclays, nclimates])
plot_NfromSOM_all = numpy.zeros([timesteps, nplots, nclays, nclimates])
plot_Nsource_all = numpy.zeros([timesteps, nplots, nclays, nclimates])
plot_Nlimit_all = numpy.zeros([timesteps, nplots, nclays, nclimates])
plot_Nlimit_acc = numpy.zeros([timesteps, nplots, nclays, nclimates])
for plotnum in range(nplots):
    for claynum in range(nclays):
        for climnum in range(nclimates):
            filename = "D:/Postdoc/CORPSE_Siya/" + str(
                6 * plotnum + 3 * claynum + climnum + 1) + "_Quarterly_data.txt"
            file = open(filename, "r")
            datastring = file.read()
            datalist = datastring.split("\n")
            for i in range(0, timesteps, 1):
                data = datalist[i].split(" ")
                plot_InorgN_all[i, plotnum, claynum, climnum] = data[0]
                plot_SAPC_all[i, plotnum, claynum, climnum] = data[1]
                plot_UnpC_all[i, plotnum, claynum, climnum] = data[2]
                plot_ECMC_all[i, plotnum, claynum, climnum] = data[3]
                plot_AMC_all[i, plotnum, claynum, climnum] = data[4]
                plot_TotN_all[i, plotnum, claynum, climnum] = data[5]
                plot_NfromNecro_all[i, plotnum, claynum, climnum] = data[6]
                plot_NfromSOM_all[i, plotnum, claynum, climnum] = data[7]
                if i==0:
                    plot_Nsource_all[i, plotnum, claynum, climnum] = float(data[6])/float(data[7])
                else:
                    plot_Nsource_all[i, plotnum, claynum, climnum] = \
                        (float(data[6])-plot_NfromNecro_all[i-1, plotnum, claynum, climnum])/(float(data[7])-plot_NfromSOM_all[i-1, plotnum, claynum, climnum])
                plot_Nlimit_acc[i, plotnum, claynum, climnum] = data[8]
                if i==0:
                    plot_Nlimit_all[i, plotnum, claynum, climnum] = 1.0
                else:
                    plot_Nlimit_all[i, plotnum, claynum, climnum] = 1/timestep * \
                         (float(data[8])-plot_Nlimit_acc[i-1, plotnum, claynum, climnum])


            plot_InorgN[plotnum, claynum, climnum] = 0.0
            for a in range(4):
                plot_InorgN[plotnum, claynum, climnum] += 0.25 * (plot_InorgN_all[-4 + a, plotnum, claynum, climnum])
            plot_SAPC[plotnum, claynum, climnum] = 0.0
            for a in range(4):
                plot_SAPC[plotnum, claynum, climnum] += 0.25 * (plot_SAPC_all[-4 + a, plotnum, claynum, climnum])
            plot_UnpC[plotnum, claynum, climnum] = 0.0
            for a in range(4):
                plot_UnpC[plotnum, claynum, climnum] += 0.25 * (plot_UnpC_all[-4 + a, plotnum, claynum, climnum])
            plot_ECMC[plotnum, claynum, climnum] = 0.0
            for a in range(4):
                plot_ECMC[plotnum, claynum, climnum] += 0.25 * (plot_ECMC_all[-4 + a, plotnum, claynum, climnum])
            plot_AMC[plotnum, claynum, climnum] = 0.0
            for a in range(4):
                plot_AMC[plotnum, claynum, climnum] += 0.25 * (plot_AMC_all[-4 + a, plotnum, claynum, climnum])
            plot_TotN[plotnum, claynum, climnum] = 0.0
            for a in range(4):
                plot_TotN[plotnum, claynum, climnum] += 0.25 * (plot_TotN_all[-4 + a, plotnum, claynum, climnum])
            plot_NfromNecro[plotnum, claynum, climnum] = 0.0
            for a in range(4):
                plot_NfromNecro[plotnum, claynum, climnum] += 0.25 * (plot_NfromNecro_all[-4 + a, plotnum, claynum, climnum])
            plot_NfromSOM[plotnum, claynum, climnum] = 0.0
            for a in range(4):
                plot_NfromSOM[plotnum, claynum, climnum] += 0.25 * (plot_NfromSOM_all[-4 + a, plotnum, claynum, climnum])

plt.figure('Inorganic N', figsize=(6, 8));
plt.clf()
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        plt.plot(ECM_pct, plot_InorgN[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))

plt.xlabel('ECM percent (%)')
plt.ylabel('Inorganic N stock (kgN/m2)')

plt.figure('Microbial C', figsize=(6, 8));
plt.clf()
ax = plt.subplot(221)
ax.set_title("SAP")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        plt.plot(ECM_pct, plot_SAPC[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        plt.xlabel('ECM percent (%)')
        plt.ylabel('SAP C (KgC/m2)')
ax = plt.subplot(222)
ax.set_title("ECM")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        plt.plot(ECM_pct, plot_ECMC[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        plt.xlabel('ECM percent (%)')
        plt.ylabel('ECM C (KgC/m2)')
ax = plt.subplot(223)
ax.set_title("AM")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        plt.plot(ECM_pct, plot_AMC[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        plt.xlabel('ECM percent (%)')
        plt.ylabel('AM C (KgC/m2)')
ax = plt.subplot(224)
ax.set_title("Total microbes")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        plt.plot(ECM_pct, plot_SAPC[:, claynum, climnum]+plot_ECMC[:, claynum, climnum]+plot_AMC[:, claynum, climnum],
                 ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        plt.xlabel('ECM percent (%)')
        plt.ylabel('Total microbial C (KgC/m2)')

plt.figure('Unprotected C', figsize=(6, 8));
plt.clf()
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        plt.plot(ECM_pct, plot_UnpC[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        plt.xlabel('ECM percent (%)')
        plt.ylabel('Unprotected C (KgC/m2)')

plt.figure('Long-term Inorganic N under 5 degree 10%Clay', figsize=(6, 8));
plt.clf()
for plotnum in range(len(ECM_pct)):
    plt.plot(plottimes,plot_InorgN_all[:,plotnum,0,0],c=cmapECM(normECM(ECM_pct[plotnum])))
    plt.xlabel('Time (year)')
    plt.ylabel('Inorganic N (KgN/m2)')
    plt.ylim([0, 0.001])

plt.figure('Long-term unprotected C under 5 degree 10%Clay', figsize=(6, 8));
plt.clf()
for plotnum in range(len(ECM_pct)):
    plt.plot(plottimes,plot_UnpC_all[:,plotnum,0,0],c=cmapECM(normECM(ECM_pct[plotnum])))
    plt.xlabel('Time (year)')
    plt.ylabel('Unprotected C (KgC/m2)')
    plt.ylim([0, 10.0])

plt.figure('Long-term total N under 5 degree 10%Clay', figsize=(6, 8));
plt.clf()
for plotnum in range(len(ECM_pct)):
    plt.plot(plottimes,plot_TotN_all[:,plotnum,0,0],c=cmapECM(normECM(ECM_pct[plotnum])))
    plt.xlabel('Time (year)')
    plt.ylabel('Total N (KgN/m2)')
    plt.ylim([0.2, 1.2])

plt.figure('Quarterly InorgN', figsize=(6, 8));
plt.clf()
time = numpy.arange(0, 1.25, 0.25)
plot_InorgN_quarterly = numpy.zeros(5)
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-5, 0, 1):
                plot_InorgN_quarterly[i + 5] = plot_InorgN_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_InorgN_quarterly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            if climnum == 0:
                plt.ylabel('InorgN (KgN/m2)')

plt.figure('Quarterly SAPC', figsize=(6, 8));
plt.clf()
time = numpy.arange(0, 1.25, 0.25)
plot_SAPC_quarterly = numpy.zeros(5)
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-5, 0, 1):
                plot_SAPC_quarterly[i + 5] = plot_SAPC_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_SAPC_quarterly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            if climnum == 0:
                plt.ylabel('SAP_C (KgC/m2)')

plt.figure('Quarterly SAP N sources from Necromass', figsize=(6, 8));
plt.clf()
time = numpy.arange(0, 1.25, 0.25)
plot_Nsource_quarterly = numpy.zeros(5)
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-5, 0, 1):
                plot_Nsource_quarterly[i + 5] = 100*plot_Nsource_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_Nsource_quarterly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            if climnum == 0:
                plt.ylabel('SAP N sources from Necromass (%)')

plt.figure('Quarterly SAP N limit status', figsize=(6, 8));
plt.clf()
time = numpy.arange(0, 1.25, 0.25)
plot_Nlimit_quarterly = numpy.zeros(5)
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-5, 0, 1):
                plot_Nlimit_quarterly[i + 5] = 100*(1-plot_Nlimit_all[i, plotnum, claynum, climnum])
            plt.plot(time, plot_Nlimit_quarterly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            if climnum == 0:
                plt.ylabel('SAP N limit status (%)')
plt.show()