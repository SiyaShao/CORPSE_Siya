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
            'Int_N_ECM': 0.0,
            'Int_N_AM': 0.0,
            'NfromNecro': 0.0,
            'NfromSOM': 0.0,
            'Nlimit': 0.0,
            'Ntransfer': 0.0,
            'Ntransfer_ECM': 0.0,
            'Ntransfer_AM': 0.0,
            'Nrootuptake': 0.0,
            'falloc_ECM': 0.0,
            'falloc_AM': 0.0}

# Set model parameters
# Note that carbon types have names, in contrast to previous version
params = {
    'vmaxref': {'Fast': 18.0, 'Slow': .5, 'Necro': 4.5},  # Relative maximum enzymatic decomp rates (year-1)
    'Ea': {'Fast': 6e3, 'Slow': 40e3, 'Necro': 6e3},
    # Activation energy (Controls temperature sensitivity via Arrhenius relationship)
    'kC': {'Fast': 0.01, 'Slow': 0.01, 'Necro': 0.01},  # Michaelis-Menton parameter
    'gas_diffusion_exp': 0.6,  # Determines suppression of decomp at high soil moisture
    'substrate_diffusion_exp': 1.5,  # Dimensionless exponent. Determines suppression of decomp at low soil moisture
    'minMicrobeC': {'SAP':1e-3, 'ECM':1e-6, 'AM':1e-6},
    # Minimum microbial biomass (fraction of total C), mycorrhizal fungi is set to be much lower
    'Tmic': {'SAP': 0.25, 'ECM': 1.0, 'AM': 1.0},  # Microbial lifetime (years)
    'et': {'SAP': 0.6, 'ECM': 0.6, 'AM': 0.6},  # Fraction of turnover not converted to CO2 (dimensionless)
    'eup': {'Fast': 0.6, 'Slow': 0.05, 'Necro': 0.6},  # Carbon uptake efficiency (dimensionless fraction)
    'tProtected': 75.0,  # Protected C turnoveir time (years)
    'protection_rate': {'Fast': 1.0, 'Slow': 0.0025, 'Necro': 1.0},  # Protected carbon formation rate (year-1)
    'new_resp_units': True,
    'frac_N_turnover_min': 0.2,
    'frac_turnover_slow': {'SAP': 0.5, 'ECM': 0.5, 'AM': 0.5},
    'nup': {'Fast': 0.9, 'Slow': 0.6, 'Necro': 0.9},
    'CN_microbe': {'SAP':8.0,'ECM':14.0,'AM':14.0},
    'max_immobilization_rate': 3.65,
    'substrate_diffusion_exp': 1.5,
    'new_resp_units': True,
    'eup_myc': {'ECM':0.5,'AM':0.5},
    'max_mining_rate': {'Fast': 0.9, 'Slow': 0.75, 'Necro': 6.75},
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
    'depth': 0.10, # 10cm, assumed for now.
    'iN_loss_rate': 30.0, # Loss rate from inorganic N pool (year-1). >1 since it takes much less than a year for it to be removed
    'N_deposition': 0.001,  # Annual nitrogen deposition 1.0gN/m2/yr
    'kG_simb': 0.3, # Half-saturation of intermediate C pool for symbiotic growth (kg C m-2)'
    'rgrowth_simb': 0.3, # Maximum growth rate of mycorrhizal fungi (kg C m-2 year-1)
    'falloc_base': 0.5, # Base allocation of NPP to mycorrhizal fungi
    'Ohorizon_transfer_rates': {'uFastC': 0.1, 'uSlowC': 0.1, 'uNecroC': 0.1, 'uFastN': 0.1, 'uSlowN': 0.1,
                                'uNecroN': 0.1},
    'ratio_ECM_scavenging': 0.025, # 'ratio of ECM inorgN scavenging rate compared to AM'
}
SOM_init['SAPN'] = SOM_init['SAPC'] / params['CN_microbe']['SAP']
SOM_init['ECMN'] = SOM_init['ECMC'] / params['CN_microbe']['ECM']
SOM_init['AMN'] = SOM_init['AMC'] / params['CN_microbe']['AM']

# ECM gradient plots
nplots = 5
nclays = 1
nclimates = 1
# Environmental conditions
# Gradient of mycorrhizal association
ECM_pct = numpy.linspace(0, 100, nplots)  # Percent ECM basal area
MAT = numpy.linspace(5, 20, nclimates)  # degrees C
clay = numpy.linspace(10, 70, nclays)  # percent clay
Croot= [0.375, 0.375, 0.375] # Deciduous: Boreal 593g/m2, Temperate 687g/m2, Tropical 1013g/m2
                   # Evergreen: Boreal 515g/m2, Temperate 836g/m2, Tropical  724g/m2, from Finér et al.,(2011)

fastfrac_AM = 0.5
fastfrac_ECM = 0.5
fastfrac_site = fastfrac_ECM * ECM_pct / 100 + fastfrac_AM * (1 - ECM_pct / 100)
litter_CN_AM = 30
litter_CN_ECM = 50
litter_CN_site = litter_CN_ECM * ECM_pct / 100 + litter_CN_AM * (1 - ECM_pct / 100)
total_inputs = 0.5  # kgC/m2

# fastfrac_site[:] = 0.1
litter_CN_site = 50

myc_ratio_NPP = 0.2  # Ratio of C transferred to mycorrhizal fungi to NPP
myc_ratio_litter = myc_ratio_NPP / (
            1 - myc_ratio_NPP)  # Ratio of C transferred to mycorrhizal fungi to litter production

inputs = {'uFastC': total_inputs * fastfrac_site,
          'uSlowC': total_inputs * (1 - fastfrac_site),
          'uFastN': total_inputs * fastfrac_site / litter_CN_site,
          'uSlowN': total_inputs * (1 - fastfrac_site) / litter_CN_site}

Ctransfer = {'ECM':0.0,'AM':0.0} # Initialize C tranfer from plants to symbiont fungi

theta = 0.5  # fraction of saturation

spinuptimes = numpy.arange(0, 2500,
                     10)  # 10-year time steps for model spinup as spinning up on finer timestep would take too long
timestep = 1/12.0  # Monthly
finaltimes = numpy.arange(0, 100 + timestep, timestep)  # Time steps to evaluat, running on quatily timestep
plottimes = finaltimes
timesteps = len(finaltimes)  # According to what is set for times in the above lines (numpy.arrange)
# The ODE solver uses an adaptive timestep but will return these time points

Inputdir = '/Users/f0068s6/Library/CloudStorage/OneDrive-McGillUniversity/Postdoc/Manuscript 2/Datasets/'
litter_filename = Inputdir + '10 sites_litter.csv'
envir_filename = Inputdir + '10 sites_envir.csv'

# Read ECM fraction and litter data
result = pandas.read_csv(litter_filename, index_col=None)
Siteofplot = result['Site']
ECMfrac = result['ECM%']
Pinefrac = result['Pine%'] + result['Eric%']
Oakfrac = result['Oak+Beech%']
LitterCN_site = result['Litter_CN']
Fastfrac_site = result['Labile%']
nplots = len(ECMfrac)

# Read environmental data
result = pandas.read_csv(envir_filename, index_col=None)
Site = result['Site']
freq_T = result['freq_T']
amplitude_T = result['amplitude_T']
phase_T = result['phase_T']
offset_T = result['offset_T']
freq_M = result['freq_M']
amplitude_M = result['amplitude_M']
phase_M = result['phase_M']
offset_M = result['offset_M']
NPP = result['NPP']
NPP_a = result['NPP_a']
NPP_x0 = result['NPP_x0']
NPP_sigma = result['NPP_sigma']
Ndep = result['Ndep']
Clay = result['Clay']

# Run the simulations
def experiment(plotnum,claynum,climnum):
    from CORPSE_deriv import sumCtypes
    print(
        'Sim {simnum:d} of {totsims:d}, Site = {site}, %ECM = {ecmpct:1.1f}, %Pine = {pinepct:1.1f}, %Oak = {oakpct:1.1f}'.format(
            simnum=nclimates*nclays*plotnum+nclimates*claynum+climnum+1, totsims=nplots*nclays*nclimates, site=Siteofplot[plotnum],
            ecmpct=ECMfrac[plotnum], pinepct=100*Pinefrac[plotnum], oakpct=100*Oakfrac[plotnum]))

    Index = numpy.where(Site == Siteofplot[plotnum])[0]
    # Use the name of site to get correct environmental information
    SoilT_params = [float(freq_T[Index]), float(amplitude_T[Index]), float(phase_T[Index]), float(offset_T[Index])]
    meanSoilT = float(offset_T[Index])
    SoilM_params = [float(freq_M[Index]), float(amplitude_M[Index]), float(phase_M[Index]), float(offset_M[Index])]
    meanSoilM = float(offset_M[Index])
    NPP_params = [float(NPP_a[Index]), float(NPP_x0[Index]), float(NPP_sigma[Index])]
    total_inputs = float(NPP[Index])
    params['N_deposition'] = float(Ndep[Index]/1000.0)
    Clay_site = float(Clay[Index])

    # litter information can be directly pinpointed using plotnum
    fastfrac_site = Fastfrac_site
    litter_CN_site = LitterCN_site[plotnum]
    inputs = {'uFastC': total_inputs * fastfrac_site,
              'uSlowC': total_inputs * (1 - fastfrac_site),
              'uFastN': total_inputs * fastfrac_site / litter_CN_site,
              'uSlowN': total_inputs * (1 - fastfrac_site) / litter_CN_site}
    Ndemand = total_inputs/litter_CN_site

    # Calculate plant Ndemand from N in litter production
    # Assuming plant N_litter balances plant N_uptake and plant not relying on roots
    # This will not be needed once the model is coupled with a plant model

    result = CORPSE_integrate.run_CORPSE_ODE(T=meanSoilT, theta=meanSoilM, Ndemand=Ndemand,
                                             inputs=dict([(k, inputs[k][plotnum]) for k in inputs]),
                                             clay=Clay_site, initvals=SOM_init, params=params,
                                             SoilT_params = SoilT_params, SoilM_params= SoilM_params,
                                             NPP_params=NPP_params, tNPP = total_inputs,
                                             times=spinuptimes, Croot=Croot[climnum], totinputs=total_inputs,
                                             litter_ECM=litter_CN_ECM, litter_AM=litter_CN_AM, totlitter=total_inputs,
                                             Pine_pct = Pinefrac[plotnum], Oak_pct = Oakfrac[plotnum],
                                             ECM_pct=ECMfrac[plotnum] / 100, runtype='Spinup')
    result = CORPSE_integrate.run_CORPSE_ODE(T=meanSoilT, theta=meanSoilM, Ndemand=Ndemand,
                                             inputs=dict([(k, inputs[k][plotnum]) for k in inputs]),
                                             clay=Clay_site, initvals=result.iloc[-1], params=params,
                                             SoilT_params=SoilT_params, SoilM_params=SoilM_params,
                                             NPP_params=NPP_params, tNPP = total_inputs,
                                             times=finaltimes, Croot=Croot[climnum], totinputs=total_inputs,
                                             litter_ECM=litter_CN_ECM, litter_AM=litter_CN_AM, totlitter=total_inputs,
                                             Pine_pct = Pinefrac[plotnum], Oak_pct = Oakfrac[plotnum],
                                             ECM_pct=ECMfrac[plotnum] / 100, runtype='Final')
    for timenum in range(timesteps):
        Outputdir = '/Users/f0068s6/Library/CloudStorage/OneDrive-McGillUniversity/Postdoc/Manuscript 2/ModelResults/'
        filename = Outputdir + str(nclimates*nclays*plotnum+nclimates*claynum + climnum + 1)+'_Monthly_data.txt'
        result.to_csv(filename)

from joblib import Parallel, delayed
output = Parallel(n_jobs=10)(
    delayed(experiment)(plotnum, claynum, climnum) for plotnum in range(nplots) for claynum in range(nclays) for
    climnum in range(nclimates))
