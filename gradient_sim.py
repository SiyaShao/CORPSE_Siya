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
            'Nlimit': 0.0,
            'Ntransfer': 0.0,
            'Ntransfer_ECM': 0.0,
            'Ntransfer_AM': 0.0,
            'Nrootuptake': 0.0}

# Set model parameters
# Note that carbon types have names, in contrast to previous version
params = {
    'vmaxref': {'Fast': 9.0, 'Slow': .5, 'Necro': 4.5},  # Relative maximum enzymatic decomp rates (year-1)
    'Ea': {'Fast': 5e3, 'Slow': 30e3, 'Necro': 3e3},
    # Activation energy (Controls temperature sensitivity via Arrhenius relationship)
    'kC': {'Fast': 0.01, 'Slow': 0.01, 'Necro': 0.01},  # Michaelis-Menton parameter
    'gas_diffusion_exp': 0.6,  # Determines suppression of decomp at high soil moisture
    'substrate_diffusion_exp': 1.5,  # Dimensionless exponent. Determines suppression of decomp at low soil moisture
    'minMicrobeC': {'SAP':1e-3, 'ECM':1e-6, 'AM':1e-6},
    # Minimum microbial biomass (fraction of total C), mycorrhizal fungi is set to be much lower
    'Tmic': {'SAP': 0.25, 'ECM': 1.0, 'AM': 0.25},  # Microbial lifetime (years)
    'et': {'SAP': 0.6, 'ECM': 0.6, 'AM': 0.6},  # Fraction of turnover not converted to CO2 (dimensionless)
    'eup': {'Fast': 0.6, 'Slow': 0.05, 'Necro': 0.6},  # Carbon uptake efficiency (dimensionless fraction)
    'tProtected': 75.0,  # Protected C turnoveir time (years)
    'protection_rate': {'Fast': 0.1, 'Slow': 0.0001, 'Necro': 1.5},  # Protected carbon formation rate (year-1)
    'new_resp_units': True,
    'frac_N_turnover_min': 0.2,
    'frac_turnover_slow': {'SAP': 0.2, 'ECM': 0.2, 'AM': 0.2},
    'nup': {'Fast': 0.9, 'Slow': 0.6, 'Necro': 0.9},
    'CN_microbe': {'SAP':8.0,'ECM':14.0,'AM':10.0},
    'max_immobilization_rate': 3.65,
    'substrate_diffusion_exp': 1.5,
    'new_resp_units': True,
    'eup_myc': {'ECM':0.5,'AM':0.5},
    'max_mining_rate': {'Fast': 1.0, 'Slow': 0.6, 'Necro': 1.0},
    # assumed to be more efficient with Slow and Necro pools, and less efficient with fast pools, compared to SAPs
    'kc_mining': 0.015, # g microbial biomass C/g substrate C, Sulman et al., (2019)
    'max_scavenging_rate': {'SAP':0.16,'ECM':0.0,'AM':0.2},
    # kgN/m2/year, AM value is from Sulman et al., (2019), assumed 80% AM capacity for SAP N immobilization, 0 for ECM
    'kc_scavenging': {'SAP':0.3,'ECM':0.3,'AM':0.3},
    # kgC/m3, AM value is from Sulman et al., (2019), assumed to be the same for other microbes
    'kc_scavenging_IN': {'SAP':0.001,'ECM':0.001,'AM':0.001},
    # kgN/m3, AM value is from Sulman et al., (2019), assumed to be the same for other microbes
    'Ea_inorgN': 37e3,  # (kJ/mol) Activation energy for immobilization of inorganic N from Sulman et al., (2019)
    'Ea_turnover': 20e3, # (kJ/mol) Activation energy for microbial turnover from Wang et al., (2013)
    'depth': 0.15, # 15cm, assumed for now.
    'iN_loss_rate': 10.0, # Loss rate from inorganic N pool (year-1). >1 since it takes much less than a year for it to be removed
    'N_deposition': 0.001,  # Annual nitrogen deposition 1.0gN/m2/yr
    'kG_simb': 0.3, # Half-saturation of intermediate C pool for symbiotic growth (kg C m-2)'
    'rgrowth_simb': 0.3, # Maximum growth rate of mycorrhizal fungi (kg C m-2 year-1)
    'Ohorizon_transfer_rates': {'uFastC': 0.1, 'uSlowC': 0.1, 'uNecroC': 0.1, 'uFastN': 0.1, 'uSlowN': 0.1,
                                'uNecroN': 0.1}
}
SOM_init['SAPN'] = SOM_init['SAPC'] / params['CN_microbe']['SAP']
SOM_init['ECMN'] = SOM_init['ECMC'] / params['CN_microbe']['ECM']
SOM_init['AMN'] = SOM_init['AMC'] / params['CN_microbe']['AM']

# ECM gradient plots
nplots = 5
nclays = 1
nclimates = 3
# Environmental conditions
# Gradient of mycorrhizal association
ECM_pct = numpy.linspace(0, 100, nplots)  # Percent ECM basal area
MAT = numpy.linspace(5, 20, nclimates)  # degrees C
clay = numpy.linspace(10, 70, nclays)  # percent clay
Croot= [0.275, 0.375, 0.45] # Deciduous: Boreal 593g/m2, Temperate 687g/m2, Tropical 1013g/m2
                   # Evergreen: Boreal 515g/m2, Temperate 836g/m2, Tropical  724g/m2, from Finér et al.,(2011)

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
timestep = 1/12.0  # Monthly
finaltimes = numpy.arange(0, 100 + timestep, timestep)  # Time steps to evaluat, running on quatily timestep
plottimes = finaltimes
timesteps = len(finaltimes)  # According to what is set for times in the above lines (numpy.arrange)
# The ODE solver uses an adaptive timestep but will return these time points


# Run the simulations
def experiment(plotnum,claynum,climnum):
    from CORPSE_deriv import sumCtypes
    print(
        'Sim {simnum:d} of {totsims:d}. %ECM = {ecmpct:1.1f}, %clay = {claypct:1.1f}, MAT = {mat:1.1f}'.format(
            simnum=nclimates*nclays*plotnum+nclimates*claynum+climnum+1, totsims=nplots*nclays*nclimates, ecmpct=ECM_pct[plotnum], claypct=clay[claynum],
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
                                             times=spinuptimes, Croot=Croot[climnum], runtype='Spinup')
    result = CORPSE_integrate.run_CORPSE_ODE(T=MAT[climnum], theta=theta, Ndemand=Ndemand,
                                             inputs=dict([(k, inputs[k][plotnum]) for k in inputs]),
                                             clay=clay[claynum], initvals=result.iloc[-1], params=params,
                                             times=finaltimes, Croot=Croot[climnum], runtype='Final')
    for timenum in range(timesteps):
        filename = str(nclimates*nclays*plotnum+nclimates*claynum + climnum + 1)+'_Monthly_data.txt'
        result.to_csv(filename)

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

# Construct the datasets for plotting
data = numpy.zeros([timesteps, nplots, nclays, nclimates])  # Store the data along all the time steps
plot_InorgN_all, plot_SAPC_all, plot_UnpC_all, plot_ECMC_all, plot_AMC_all, plot_TotN_all, plot_NfromNecro_all, \
plot_NfromSOM_all, plot_Nsource_all, plot_Nlimit_all, plot_Nlimit_acc = data.copy(), data.copy(), data.copy(), \
                                                                        data.copy(), data.copy(), data.copy(), \
                                                                        data.copy(), data.copy(), data.copy(), \
                                                                        data.copy(), data.copy()
plot_IntN_all, plot_Nrootuptake_all, plot_Ntransfer_all, plot_Ntransfer_ECM_all, plot_Ntransfer_AM_all, \
plot_Nrootuptake_acc, plot_Ntransfer_acc, plot_Ntransfer_ECM_acc, plot_Ntransfer_AM_acc = data.copy(), data.copy(), \
                                                                                          data.copy(), data.copy(), \
                                                                                          data.copy(), data.copy(), \
                                                                                          data.copy(),data.copy(), \
                                                                                          data.copy()
data = numpy.zeros([nplots, nclays, nclimates])  # Store the annual data of the last year
plot_InorgN, plot_SAPC, plot_UnpC, plot_ECMC, plot_AMC, plot_TotN, plot_NfromNecro, plot_NfromSOM, plot_Nsource, \
plot_Nlimit = data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), \
              data.copy(), data.copy()
plot_Nrootuptake, plot_Ntransfer, plot_Ntransfer_ECM, plot_Ntransfer_AM = data.copy(), data.copy(), data.copy(), \
                                                                          data.copy()

for plotnum in range(nplots):
    for claynum in range(nclays):
        for climnum in range(nclimates):
            filename = "D:/Postdoc/CORPSE_Siya/" + str(nclimates * nclays * plotnum + nclimates * claynum + climnum + 1)\
                       + '_Monthly_data.txt'
            result = pandas.read_csv(filename)
            plot_InorgN_all[:, plotnum, claynum, climnum] = result['inorganicN']
            plot_SAPC_all[:, plotnum, claynum, climnum] = result['SAPC']
            plot_UnpC_all[:, plotnum, claynum, climnum] = result['uFastC']+result['uSlowC']+result['uNecroC']
            plot_ECMC_all[:, plotnum, claynum, climnum] = result['ECMC']
            plot_AMC_all[:, plotnum, claynum, climnum] = result['AMC']
            plot_TotN_all[:, plotnum, claynum, climnum] = result['uFastN']+result['uSlowN']+result['uNecroN']+\
                                                          result['pFastN']+result['pSlowN']+result['pNecroN']
            plot_NfromNecro_all[:, plotnum, claynum, climnum] = result['NfromNecro']
            plot_NfromSOM_all[:, plotnum, claynum, climnum] = result['NfromSOM']
            plot_Nlimit_acc[:, plotnum, claynum, climnum] = result['Nlimit']
            plot_IntN_all[:, plotnum, claynum, climnum] = result['Int_N']
            plot_Ntransfer_acc[:, plotnum, claynum, climnum] = result['Ntransfer']
            plot_Ntransfer_ECM_acc[:, plotnum, claynum, climnum] = result['Ntransfer_ECM']
            plot_Ntransfer_AM_acc[:, plotnum, claynum, climnum] = result['Ntransfer_AM']
            plot_Nrootuptake_acc[:, plotnum, claynum, climnum] = result['Nrootuptake']

            for i in range(0, timesteps, 1):
                if i==0:
                    plot_Nsource_all[i, plotnum, claynum, climnum] = plot_NfromNecro_all[i, plotnum, claynum, climnum]/\
                                                                     plot_NfromSOM_all[i, plotnum, claynum, climnum]
                else:
                    plot_Nsource_all[i, plotnum, claynum, climnum] = \
                        (plot_NfromNecro_all[i, plotnum, claynum, climnum]-plot_NfromNecro_all[i-1, plotnum, claynum, climnum])/\
                        (plot_NfromSOM_all[i, plotnum, claynum, climnum]-plot_NfromSOM_all[i-1, plotnum, claynum, climnum])
                if i==0:
                    plot_Nlimit_all[i, plotnum, claynum, climnum] = 1.0
                    plot_Ntransfer_all[i, plotnum, claynum, climnum] = plot_Ntransfer_acc[i, plotnum, claynum, climnum]
                    plot_Ntransfer_ECM_all[i, plotnum, claynum, climnum] = plot_Ntransfer_ECM_acc[i, plotnum, claynum, climnum]
                    plot_Ntransfer_AM_all[i, plotnum, claynum, climnum] = plot_Ntransfer_AM_acc[i, plotnum, claynum, climnum]
                    plot_Nrootuptake_all[i, plotnum, claynum, climnum] = plot_Nrootuptake_acc[i, plotnum, claynum, climnum]
                else:
                    plot_Nlimit_all[i, plotnum, claynum, climnum] = 1/timestep * \
                         (plot_Nlimit_acc[i, plotnum, claynum, climnum]-plot_Nlimit_acc[i-1, plotnum, claynum, climnum])
                    plot_Ntransfer_all[i, plotnum, claynum, climnum] = 1/timestep * \
                         (plot_Ntransfer_acc[i, plotnum, claynum, climnum]-plot_Ntransfer_acc[i-1, plotnum, claynum, climnum])
                    plot_Ntransfer_ECM_all[i, plotnum, claynum, climnum] = 1 / timestep * \
                                                                    (plot_Ntransfer_ECM_acc[i, plotnum, claynum, climnum] -
                                                                     plot_Ntransfer_ECM_acc[i - 1, plotnum, claynum, climnum])
                    plot_Ntransfer_AM_all[i, plotnum, claynum, climnum] = 1 / timestep * \
                                                                    (plot_Ntransfer_AM_acc[i, plotnum, claynum, climnum] -
                                                                     plot_Ntransfer_AM_acc[i - 1, plotnum, claynum, climnum])
                    plot_Nrootuptake_all[i, plotnum, claynum, climnum] = 1 / timestep * \
                                                                    (plot_Nrootuptake_acc[i, plotnum, claynum, climnum] -
                                                                     plot_Nrootuptake_acc[i - 1, plotnum, claynum, climnum])

            for a in range(int(1/timestep)):
                plot_InorgN[plotnum, claynum, climnum] += timestep * (plot_InorgN_all[-int(1/timestep) + a, plotnum, claynum, climnum])
                plot_SAPC[plotnum, claynum, climnum] += timestep * (plot_SAPC_all[-int(1/timestep) + a, plotnum, claynum, climnum])
                plot_UnpC[plotnum, claynum, climnum] += timestep * (plot_UnpC_all[-int(1/timestep) + a, plotnum, claynum, climnum])
                plot_ECMC[plotnum, claynum, climnum] += timestep * (plot_ECMC_all[-int(1/timestep) + a, plotnum, claynum, climnum])
                plot_AMC[plotnum, claynum, climnum] += timestep * (plot_AMC_all[-int(1/timestep) + a, plotnum, claynum, climnum])
                plot_TotN[plotnum, claynum, climnum] += timestep * (plot_TotN_all[-int(1/timestep) + a, plotnum, claynum, climnum])
                plot_NfromNecro[plotnum, claynum, climnum] += timestep * (plot_NfromNecro_all[-int(1/timestep) + a, plotnum, claynum, climnum])
                plot_NfromSOM[plotnum, claynum, climnum] += timestep * (plot_NfromSOM_all[-int(1/timestep) + a, plotnum, claynum, climnum])
                plot_Ntransfer[plotnum, claynum, climnum] += timestep * (plot_Ntransfer_all[-int(1/timestep) + a, plotnum, claynum, climnum])
                plot_Ntransfer_ECM[plotnum, claynum, climnum] += timestep * (plot_Ntransfer_ECM_all[-int(1/timestep) + a, plotnum, claynum, climnum])
                plot_Ntransfer_AM[plotnum, claynum, climnum] += timestep * (plot_Ntransfer_AM_all[-int(1/timestep) + a, plotnum, claynum, climnum])
                plot_Nrootuptake[plotnum, claynum, climnum] += timestep * (plot_Nrootuptake_all[-int(1 / timestep) + a, plotnum, claynum, climnum])

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

plt.figure('Monthly InorgN', figsize=(6, 8));
plt.clf()
time = numpy.arange(0, 1+timestep, timestep)
plot_InorgN_Monthly = numpy.zeros(1+int(1/timestep))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_InorgN_Monthly[i + 1+int(1/timestep)] = plot_InorgN_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_InorgN_Monthly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            if climnum == 0:
                plt.ylabel('InorgN (KgN/m2)')

plt.figure('Monthly microbes', figsize=(6, 8));
plt.clf()
ax = plt.subplot(131)
ax.set_title("Monthly SAP")
time = numpy.arange(0, 1+timestep, timestep)
plot_SAPC_Monthly = numpy.zeros(1+int(1/timestep))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_SAPC_Monthly[i + 1+int(1/timestep)] = plot_SAPC_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_SAPC_Monthly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            plt.ylabel('SAP_C (KgC/m2)')
ax = plt.subplot(132)
ax.set_title("Monthly ECMC")
time = numpy.arange(0, 1+timestep, timestep)
plot_ECMC_Monthly = numpy.zeros(1+int(1/timestep))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_ECMC_Monthly[i + 1+int(1/timestep)] = plot_ECMC_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_ECMC_Monthly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            plt.ylabel('ECM_C (KgC/m2)')
ax = plt.subplot(133)
ax.set_title("Monthly AM")
time = numpy.arange(0, 1+timestep, timestep)
plot_AMC_Monthly = numpy.zeros(1+int(1/timestep))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_AMC_Monthly[i + 1+int(1/timestep)] = plot_AMC_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_AMC_Monthly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            plt.ylabel('AM_C (KgC/m2)')

# plt.figure('Monthly SAP N sources from Necromass', figsize=(6, 8));
# plt.clf()
# time = numpy.arange(0, 1+timestep, timestep)
# plot_Nsource_Monthly = numpy.zeros((1+int(1/timestep)))
# for plotnum in range(len(ECM_pct)):
#     for claynum in range(len(clay)):
#         for climnum in range(len(MAT)):
#             ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
#             ax.set_title(str(MAT[climnum])+"°C")
#             for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
#                 plot_Nsource_Monthly[i + 1+int(1/timestep)] = 100*plot_Nsource_all[i, plotnum, claynum, climnum]
#             plt.plot(time, plot_Nsource_Monthly[:], ms=4, marker=markers[claynum],
#                      c=cmapECM(normECM(ECM_pct[plotnum])),
#                      label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
#             plt.xlabel('Time (year)')
#             if climnum == 0:
#                 plt.ylabel('SAP N sources from Necromass (%)')
#
# plt.figure('Monthly SAP N limit status', figsize=(6, 8));
# plt.clf()
# time = numpy.arange(0, 1+timestep, timestep)
# plot_Nlimit_Monthly = numpy.zeros(1+int(1/timestep))
# for plotnum in range(len(ECM_pct)):
#     for claynum in range(len(clay)):
#         for climnum in range(len(MAT)):
#             ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
#             ax.set_title(str(MAT[climnum])+"°C")
#             for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
#                 plot_Nlimit_Monthly[i + 1+int(1/timestep)] = 100*(1-plot_Nlimit_all[i, plotnum, claynum, climnum])
#             plt.plot(time, plot_Nlimit_Monthly[:], ms=4, marker=markers[claynum],
#                      c=cmapECM(normECM(ECM_pct[plotnum])),
#                      label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
#             plt.xlabel('Time (year)')
#             if climnum == 0:
#                 plt.ylabel('SAP N limit status (%)')

plt.figure('Monthly Intermediate N', figsize=(6, 8));
plt.clf()
time = numpy.arange(0, 1+timestep, timestep)
plot_IntN_Monthly = numpy.zeros(1+int(1/timestep))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_IntN_Monthly[i + 1+int(1/timestep)] = plot_IntN_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_IntN_Monthly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            if climnum == 0:
                plt.ylabel('Monthly Intermediate N')

plt.figure('SAP N sources from Necromass (%)', figsize=(6, 8));
plt.clf()
time = numpy.arange(0, 1+timestep, timestep)
plot_Nsource_Monthly = numpy.zeros((1+int(1/timestep)))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_Nsource_Monthly[i + 1+int(1/timestep)] = 100*plot_Nsource_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_Nsource_Monthly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            if climnum == 0:
                plt.ylabel('SAP N sources from Necromass (%)')


plt.figure('Monthly MYC transfer', figsize=(6, 8));
plt.clf()
ax = plt.subplot(221)
ax.set_title("Monthly total MYC transfer")
time = numpy.arange(0, 1+timestep, timestep)
plot_Ntransfer_Monthly = numpy.zeros(1+int(1/timestep))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            # ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            # ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_Ntransfer_Monthly[i + 1+int(1/timestep)] = plot_Ntransfer_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_Ntransfer_Monthly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            if climnum == 0:
                plt.ylabel('MYC N transfer')

ax = plt.subplot(222)
ax.set_title("Monthly ECM transfer")
time = numpy.arange(0, 1+timestep, timestep)
plot_Ntransfer_ECM_Monthly = numpy.zeros(1+int(1/timestep))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            # ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            # ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_Ntransfer_ECM_Monthly[i + 1+int(1/timestep)] = plot_Ntransfer_ECM_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_Ntransfer_ECM_Monthly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            if climnum == 0:
                plt.ylabel('ECM N transfer')

ax = plt.subplot(223)
ax.set_title("Monthly AM transfer")
time = numpy.arange(0, 1+timestep, timestep)
plot_Ntransfer_AM_Monthly = numpy.zeros(1+int(1/timestep))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            # ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            # ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_Ntransfer_AM_Monthly[i + 1+int(1/timestep)] = plot_Ntransfer_AM_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_Ntransfer_AM_Monthly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            if climnum == 0:
                plt.ylabel('AM N transfer')
ax = plt.subplot(224)
ax.set_title("Monthly Root uptake")
time = numpy.arange(0, 1+timestep, timestep)
plot_Nrootuptake_Monthly = numpy.zeros(1+int(1/timestep))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            # ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            # ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_Nrootuptake_Monthly[i + 1+int(1/timestep)] = plot_Nrootuptake_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_Nrootuptake_Monthly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            if climnum == 0:
                plt.ylabel('N_root uptake')


plt.figure('N uptake', figsize=(6, 8));
plt.clf()
ax = plt.subplot(321)
ax.set_title("Annual mycorrhizal transfer")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        plt.plot(ECM_pct, plot_Ntransfer[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        # plt.xlabel('ECM percent (%)')
        plt.ylabel('Annual mycorrhizal transfer')
ax = plt.subplot(322)
ax.set_title("Annual ECM transfer")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        plt.plot(ECM_pct, plot_Ntransfer_ECM[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        # plt.xlabel('ECM percent (%)')
        plt.ylabel('Annual ECM transfer')
ax = plt.subplot(323)
ax.set_title("Annual AM transfer")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        print(climnum, plot_Ntransfer_AM[:, claynum, climnum])
        plt.plot(ECM_pct, plot_Ntransfer_AM[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        # plt.xlabel('ECM percent (%)')
        plt.ylabel('Annual AM transfer')
ax = plt.subplot(324)
ax.set_title("Root N uptake")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        print(climnum, plot_Nrootuptake[:, claynum, climnum])
        plt.plot(ECM_pct, plot_Nrootuptake[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        plt.xlabel('ECM percent (%)')
        plt.ylabel('Root N uptake')
ax = plt.subplot(325)
ax.set_title("Total N uptake")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        print(climnum,plot_Nrootuptake[:, claynum, climnum]+plot_Ntransfer[:, claynum, climnum])
        plt.plot(ECM_pct, plot_Nrootuptake[:, claynum, climnum]+plot_Ntransfer[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        plt.xlabel('ECM percent (%)')
        plt.ylabel('Total N uptake')

plt.show()