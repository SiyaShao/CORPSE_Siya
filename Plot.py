# Plot the results
import pandas, numpy
import matplotlib.pyplot as plt
import seaborn as sns

# Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = numpy.polyfit(x, y, degree)
     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    correlation = numpy.corrcoef(x, y)[0,1]

     # r
    results['correlation'] = correlation
     # r-squared
    results['determination'] = correlation**2

    return results

Inputdir = '/Users/f0068s6/Library/CloudStorage/OneDrive-McGillUniversity/Postdoc/Manuscript 2/Datasets/'
litter_filename = Inputdir + '10 sites_litter.csv'
Soil_filename = Inputdir + '10 sites_SoilCN.csv'
envir_filename = Inputdir + '10 sites_envir.csv'

result = pandas.read_csv(litter_filename, index_col=None)
result_soil = pandas.read_csv(Soil_filename, index_col=None)
result_env = pandas.read_csv(envir_filename, index_col=None)
Siteofplot = result['Site']
ECMfrac = result['ECM%']
obs_TotCN = result_soil['SoilCN']
obs_UnpCN = result_soil['UnproCN']
obs_UnpC = result_soil['UnpC']
obs_frac_pro = result_soil['Frac_pro']
BD = result_env['BD']
Site_env = result_env['Site']

# ECM gradient plots
nplots = len(ECMfrac)
nclays = 1
nclimates = 1
# Environmental conditions
# Gradient of mycorrhizal association
ECM_pct = ECMfrac  # Percent ECM basal area
MAT = numpy.linspace(5, 20, nclimates)  # degrees C
clay = numpy.linspace(10, 70, nclays)  # percent clay
timestep = 1/12.0  # Monthly
finaltimes = numpy.arange(0, 100 + timestep, timestep)  # Time steps to evaluat, running on quatily timestep
plottimes = finaltimes
timesteps = len(finaltimes)  # According to what is set for times in the above lines (numpy.arrange)

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
plot_UnpN_all = data.copy()
plot_IntN_AM_all, plot_IntN_ECM_all, plot_Nrootuptake_all, plot_Ntransfer_all, plot_Ntransfer_ECM_all, plot_Ntransfer_AM_all, \
plot_Nrootuptake_acc, plot_Ntransfer_acc, plot_Ntransfer_ECM_acc, plot_Ntransfer_AM_acc, \
plot_falloc_ECM_all, plot_falloc_ECM_acc = data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), \
                                   data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), data.copy()
plot_IntECMC_all, plot_TotC_all, plot_falloc_AM_all, plot_falloc_AM_acc, plot_IntAMC_all = data.copy(), data.copy(), data.copy(), \
                                                                                           data.copy(), data.copy()
data = numpy.zeros([nplots, nclays, nclimates])  # Store the annual data of the last year
plot_InorgN, plot_SAPC, plot_UnpC, plot_ECMC, plot_AMC, plot_TotN, plot_NfromNecro, plot_NfromSOM, plot_Nsource, \
plot_Nlimit, plot_TotC = data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), \
                         data.copy(), data.copy(), data.copy(), data.copy()
plot_Nrootuptake, plot_Ntransfer, plot_Ntransfer_ECM, plot_Ntransfer_AM, plot_falloc_ECM = data.copy(), data.copy(), \
                                                                                       data.copy(), data.copy(), data.copy()
plot_falloc_AM,plot_falloc_ECM, plot_IntECMC, plot_IntAMC, plot_UnpN = data.copy(),data.copy(),data.copy(),data.copy(),data.copy()

for plotnum in range(nplots):
    for claynum in range(nclays):
        for climnum in range(nclimates):
            Resultsdir = '/Users/f0068s6/Library/CloudStorage/OneDrive-McGillUniversity/Postdoc/Manuscript 2/Datasets/'
            Resultsdir = '/Users/f0068s6/Library/CloudStorage/OneDrive-McGillUniversity/Postdoc/Manuscript 2/ModelResults/'
            filename = Resultsdir + str(nclimates * nclays * plotnum + nclimates * claynum + climnum + 1)\
                       + '_Monthly_data.txt'
            result = pandas.read_csv(filename)
            plot_InorgN_all[:, plotnum, claynum, climnum] = result['inorganicN']
            plot_SAPC_all[:, plotnum, claynum, climnum] = result['SAPC']
            plot_UnpC_all[:, plotnum, claynum, climnum] = result['uFastC']+result['uSlowC']+result['uNecroC']
            plot_UnpN_all[:, plotnum, claynum, climnum] = result['uFastN'] + result['uSlowN'] + result['uNecroN']
            plot_TotC_all[:, plotnum, claynum, climnum] = plot_UnpC_all[:, plotnum, claynum, climnum]+\
                                                          result['pFastC']+result['pSlowC']+result['pNecroC']
            plot_ECMC_all[:, plotnum, claynum, climnum] = result['ECMC']
            plot_AMC_all[:, plotnum, claynum, climnum] = result['AMC']
            plot_TotN_all[:, plotnum, claynum, climnum] = result['uFastN']+result['uSlowN']+result['uNecroN']+\
                                                          result['pFastN']+result['pSlowN']+result['pNecroN']
            plot_NfromNecro_all[:, plotnum, claynum, climnum] = result['NfromNecro']
            plot_NfromSOM_all[:, plotnum, claynum, climnum] = result['NfromSOM']
            plot_Nlimit_acc[:, plotnum, claynum, climnum] = result['Nlimit']
            plot_IntN_AM_all[:, plotnum, claynum, climnum] = result['Int_N_AM']
            plot_IntN_ECM_all[:, plotnum, claynum, climnum] = result['Int_N_ECM']
            plot_Ntransfer_acc[:, plotnum, claynum, climnum] = result['Ntransfer']
            plot_Ntransfer_ECM_acc[:, plotnum, claynum, climnum] = result['Ntransfer_ECM']
            plot_Ntransfer_AM_acc[:, plotnum, claynum, climnum] = result['Ntransfer_AM']
            plot_Nrootuptake_acc[:, plotnum, claynum, climnum] = result['Nrootuptake']
            plot_falloc_ECM_acc[:, plotnum, claynum, climnum] = result['falloc_ECM']
            plot_falloc_AM_acc[:, plotnum, claynum, climnum] = result['falloc_AM']
            plot_IntECMC_all[:, plotnum, claynum, climnum] = result['Int_ECMC']
            plot_IntAMC_all[:, plotnum, claynum, climnum] = result['Int_AMC']

            for i in range(0, timesteps, 1):
                if i==0:
                    plot_Nsource_all[i, plotnum, claynum, climnum] = plot_NfromNecro_all[i, plotnum, claynum, climnum]/\
                                                                     plot_NfromSOM_all[i, plotnum, claynum, climnum]
                else:
                    plot_Nsource_all[i, plotnum, claynum, climnum] = \
                        (plot_NfromNecro_all[i, plotnum, claynum, climnum]-plot_NfromNecro_all[i-1, plotnum, claynum, climnum])/\
                        (plot_NfromSOM_all[i, plotnum, claynum, climnum]-plot_NfromSOM_all[i-1, plotnum, claynum, climnum])
                if i==0:
                    plot_Nlimit_all[i, plotnum, claynum, climnum] = plot_Nlimit_acc[i, plotnum, claynum, climnum]
                    plot_Ntransfer_all[i, plotnum, claynum, climnum] = plot_Ntransfer_acc[i, plotnum, claynum, climnum]
                    plot_Ntransfer_ECM_all[i, plotnum, claynum, climnum] = plot_Ntransfer_ECM_acc[i, plotnum, claynum, climnum]
                    plot_Ntransfer_AM_all[i, plotnum, claynum, climnum] = plot_Ntransfer_AM_acc[i, plotnum, claynum, climnum]
                    plot_Nrootuptake_all[i, plotnum, claynum, climnum] = plot_Nrootuptake_acc[i, plotnum, claynum, climnum]
                    plot_falloc_ECM_all[i, plotnum, claynum, climnum] = plot_falloc_ECM_acc[i, plotnum, claynum, climnum]
                    plot_falloc_AM_all[i, plotnum, claynum, climnum] = plot_falloc_AM_acc[i, plotnum, claynum, climnum]
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
                    plot_falloc_ECM_all[i, plotnum, claynum, climnum] = 1/timestep * \
                         (plot_falloc_ECM_acc[i, plotnum, claynum, climnum]-plot_falloc_ECM_acc[i-1, plotnum, claynum, climnum])
                    plot_falloc_AM_all[i, plotnum, claynum, climnum] = 1/timestep * \
                         (plot_falloc_AM_acc[i, plotnum, claynum, climnum]-plot_falloc_AM_acc[i-1, plotnum, claynum, climnum])

            average_time = 10*int(1/timestep)
            factor_time = 1/average_time
            for a in range(average_time):
                plot_InorgN[plotnum, claynum, climnum] += factor_time * (plot_InorgN_all[-average_time + a, plotnum, claynum, climnum])
                plot_SAPC[plotnum, claynum, climnum] += factor_time * (plot_SAPC_all[-average_time + a, plotnum, claynum, climnum])
                plot_UnpC[plotnum, claynum, climnum] += factor_time * (plot_UnpC_all[-average_time + a, plotnum, claynum, climnum])
                plot_UnpN[plotnum, claynum, climnum] += factor_time * (plot_UnpN_all[-average_time + a, plotnum, claynum, climnum])
                plot_ECMC[plotnum, claynum, climnum] += factor_time * (plot_ECMC_all[-average_time + a, plotnum, claynum, climnum])
                plot_AMC[plotnum, claynum, climnum] += factor_time * (plot_AMC_all[-average_time + a, plotnum, claynum, climnum])
                plot_TotN[plotnum, claynum, climnum] += factor_time * (plot_TotN_all[-average_time + a, plotnum, claynum, climnum])
                plot_TotC[plotnum, claynum, climnum] += factor_time * (plot_TotC_all[-average_time + a, plotnum, claynum, climnum])
                plot_NfromNecro[plotnum, claynum, climnum] += factor_time * (plot_NfromNecro_all[-average_time + a, plotnum, claynum, climnum])
                plot_NfromSOM[plotnum, claynum, climnum] += factor_time * (plot_NfromSOM_all[-average_time + a, plotnum, claynum, climnum])
                plot_Ntransfer[plotnum, claynum, climnum] += factor_time * (plot_Ntransfer_all[-average_time + a, plotnum, claynum, climnum])
                plot_Ntransfer_ECM[plotnum, claynum, climnum] += factor_time * (plot_Ntransfer_ECM_all[-average_time + a, plotnum, claynum, climnum])
                plot_Ntransfer_AM[plotnum, claynum, climnum] += factor_time * (plot_Ntransfer_AM_all[-average_time + a, plotnum, claynum, climnum])
                plot_Nrootuptake[plotnum, claynum, climnum] += factor_time * (plot_Nrootuptake_all[-average_time + a, plotnum, claynum, climnum])
                plot_falloc_AM[plotnum, claynum, climnum] += factor_time * (plot_falloc_AM_all[-average_time + a, plotnum, claynum, climnum])
                plot_falloc_ECM[plotnum, claynum, climnum] += factor_time * (plot_falloc_ECM_all[-average_time+ a, plotnum, claynum, climnum])
                plot_IntECMC[plotnum, claynum, climnum] += factor_time * (plot_IntECMC_all[-average_time + a, plotnum, claynum, climnum])
                plot_IntAMC[plotnum, claynum, climnum] += factor_time * (plot_IntAMC_all[-average_time + a, plotnum, claynum, climnum])

plt.figure('Total soil C:N', figsize=(6, 8));
plt.clf()
TotCNratios = numpy.zeros([nplots])
TotCNratios = plot_TotC[:, claynum, climnum]/plot_TotN[:, claynum, climnum]
places = ['NH', 'GA', 'WI', 'IL', 'HF', 'LDW', 'SCBI', 'SRC', 'TRC', 'WLF']
palette = sns.color_palette("rainbow", len(places))
for i in range(3):
    for j in range(4):
        if i==2 & j>1:
            break
        else:
            ax = plt.subplot2grid((3, 4), (i, j))
            placenum = i*4+j
            Index = numpy.where(Siteofplot == places[placenum])[0]
            plot_ECMpct = ECM_pct[Index]
            plot_TotCNratios = TotCNratios[Index]
            plot_obs_TotCNratios = obs_TotCN[Index]
            plt.scatter(plot_ECMpct, plot_TotCNratios, s=120, marker='o',
                     facecolors='none', edgecolors=palette[placenum], alpha = 0.7,
                     label=places[placenum]+'_modeled')
            plt.scatter(plot_ECMpct, plot_obs_TotCNratios, s=120, marker='*',
                        facecolors='none', edgecolors=palette[placenum], alpha=0.7,
                        label=places[placenum]+'_observed')
            ax.set_xlabel('ECM percent (%)', fontsize = 8)
            plt.xticks(fontsize=8)
            ax.set_ylabel('Total soil C:N ratios', fontsize = 8)
            plt.yticks(fontsize=8)
            plt.legend(frameon=False, prop={"size": 8})
# for claynum in range(len(clay)):
#     for climnum in range(len(MAT)):
#         plt.plot(ECM_pct, plot_TotC[:, claynum, climnum]/plot_TotN[:, claynum, climnum], ms=4, marker=markers[claynum],
#                  c=cmap(norm(MAT[climnum])),
#                  label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
#         plt.xlabel('ECM percent (%)')
plt.figure('Scatterplot of total soil C:N', figsize=(6, 6));
plt.scatter(obs_TotCN, TotCNratios, s=120, marker='o', facecolors='none', edgecolors='b', alpha = 0.7)
plt.xlim([10, 35])
plt.ylim([10, 35])
plt.xlabel('Observed total soil C:N ratios')
plt.ylabel('Simulated total soil C:N ratios')
z = numpy.polyfit(obs_TotCN, TotCNratios, 1)
p = numpy.poly1d(z)
plt.plot(obs_TotCN, p(obs_TotCN), "r-")
stats_results = polyfit(obs_TotCN, TotCNratios, 1)
r_squared = stats_results['determination']
plt.text(25, 25, '$R^2$ = {R_squared:2.3f}'.format(R_squared=r_squared), fontsize=15)

plt.figure('Unportected soil C:N', figsize=(6, 8));
plt.clf()
UnpCNratios = numpy.zeros([nplots])
UnpCNratios = plot_UnpC[:, claynum, climnum]/plot_UnpN[:, claynum, climnum]
places = ['NH', 'GA', 'WI', 'IL', 'HF', 'LDW', 'SCBI', 'SRC', 'TRC', 'WLF']
palette = sns.color_palette("rainbow", len(places))
for i in range(3):
    for j in range(4):
        if i==2 & j>1:
            break
        else:
            ax = plt.subplot2grid((3, 4), (i, j))
            placenum = i*4+j
            Index = numpy.where(Siteofplot == places[placenum])[0]
            plot_ECMpct = ECM_pct[Index]
            plot_UnpCNratios = UnpCNratios[Index]
            plot_obs_UnpCN = obs_UnpCN[Index]
            plt.scatter(plot_ECMpct, plot_UnpCNratios, s=120, marker='o',
                     facecolors='none', edgecolors=palette[placenum], alpha = 0.7,
                     label=places[placenum]+'_modeled')
            plt.scatter(plot_ECMpct, plot_obs_UnpCN, s=120, marker='*',
                        facecolors='none', edgecolors=palette[placenum], alpha=0.7,
                        label=places[placenum]+'_observed')
            ax.set_xlabel('ECM percent (%)', fontsize = 8)
            plt.xticks(fontsize=8)
            ax.set_ylabel('Unprotected soil C:N ratios', fontsize = 8)
            plt.yticks(fontsize=8)
            plt.legend(frameon=False, prop={"size": 8})
plt.figure('Scatterplot of unprotected soil C:N', figsize=(6, 6));
plt.scatter(obs_UnpCN, UnpCNratios, s=120, marker='o', facecolors='none', edgecolors='b', alpha = 0.7)
plt.xlim([0, 70])
plt.ylim([0, 70])
plt.xlabel('Observed unprotected soil C:N ratios')
plt.ylabel('Simulated unprotected soil C:N ratios')
z = numpy.polyfit(obs_UnpCN, UnpCNratios, 1)
p = numpy.poly1d(z)
ax = plt.plot(obs_UnpCN, p(obs_UnpCN), "r-")
stats_results = polyfit(obs_UnpCN, UnpCNratios, 1)
r_squared = stats_results['determination']
plt.text(40, 40, '$R^2$ = {R_squared:2.3f}'.format(R_squared=r_squared), fontsize=15)

plt.figure('Fraction of protected soil', figsize=(6, 8));
plt.clf()
Frac_proC = numpy.zeros([nplots])
Frac_proC = (plot_TotC[:, claynum, climnum]-plot_UnpC[:, claynum, climnum])/plot_TotC[:, claynum, climnum]
places = ['NH', 'GA', 'WI', 'IL', 'HF', 'LDW', 'SCBI', 'SRC', 'TRC', 'WLF']
palette = sns.color_palette("rainbow", len(places))
for i in range(3):
    for j in range(4):
        if i==2 & j>1:
            break
        else:
            ax = plt.subplot2grid((3, 4), (i, j))
            placenum = i*4+j
            Index = numpy.where(Siteofplot == places[placenum])[0]
            plot_ECMpct = ECM_pct[Index]
            plot_Frac_proC = 100*Frac_proC[Index]
            plot_obs_frac_pro = 100*obs_frac_pro[Index]
            plt.scatter(plot_ECMpct, plot_Frac_proC, s=120, marker='o',
                     facecolors='none', edgecolors=palette[placenum], alpha = 0.7,
                     label=places[placenum]+'_modeled')
            plt.scatter(plot_ECMpct, plot_obs_frac_pro, s=120, marker='*',
                        facecolors='none', edgecolors=palette[placenum], alpha=0.7,
                        label=places[placenum]+'_observed')
            ax.set_xlabel('ECM percent (%)', fontsize = 8)
            plt.xticks(fontsize=8)
            ax.set_ylabel('Fraction of protected soil (%)', fontsize = 8)
            plt.yticks(fontsize=8)
            plt.legend(frameon=False, prop={"size": 8})
plt.figure('Scatterplot of fractions of protected soil', figsize=(6, 6));
plt.scatter(obs_frac_pro*100, Frac_proC*100, s=120, marker='o', facecolors='none', edgecolors='b', alpha = 0.7)
plt.xlim([0, 100])
plt.ylim([0, 100])
plt.xlabel('Observed fractions of protected soil')
plt.ylabel('Simulated fractions of protected soil')
z = numpy.polyfit(obs_frac_pro*100, Frac_proC*100, 1)
p = numpy.poly1d(z)
plt.plot(obs_frac_pro*100, p(obs_frac_pro*100), "r-")
stats_results = polyfit(obs_frac_pro*100, Frac_proC*100, 1)
r_squared = stats_results['determination']
plt.text(40, 60, '$R^2$ = {R_squared:2.3f}'.format(R_squared=r_squared), fontsize=15)

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
# ax.set_title("SAP")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        plt.plot(ECM_pct, plot_SAPC[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        plt.xlabel('ECM percent (%)')
        plt.ylabel('SAP C (KgC/m2)')
ax = plt.subplot(222)
# ax.set_title("ECM")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        plt.plot(ECM_pct, plot_ECMC[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        plt.xlabel('ECM percent (%)')
        plt.ylabel('ECM C (KgC/m2)')
ax = plt.subplot(223)
# ax.set_title("AM")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        plt.plot(ECM_pct, plot_AMC[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        plt.xlabel('ECM percent (%)')
        plt.ylabel('AM C (KgC/m2)')
ax = plt.subplot(224)
# ax.set_title("Total microbes")
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
for i in range(3):
    for j in range(4):
        if i==2 & j>1:
            break
        else:
            ax = plt.subplot2grid((3, 4), (i, j))
            placenum = i*4+j
            Index = numpy.where(Siteofplot == places[placenum])[0]
            Index_env = numpy.where(Site_env == places[placenum])[0]
            plot_ECMpct = ECM_pct[Index]
            plot_obs_UnpC = obs_UnpC[Index]
            if placenum>3:
                obs_UnpC[Index] = obs_UnpC[Index]*100
                plot_obs_UnpC = plot_obs_UnpC*100 # Converting units for Craig data
            print(float(BD[Index_env]))
            plt.scatter(plot_ECMpct, plot_UnpC[Index, 0, 0]/float(BD[Index_env])/1.0, s=120, marker='o',
                     facecolors='none', edgecolors=palette[placenum], alpha = 0.7,
                     label=places[placenum]+'_modeled')
            plt.scatter(plot_ECMpct, plot_obs_UnpC, s=120, marker='*',
                        facecolors='none', edgecolors=palette[placenum], alpha=0.7,
                        label=places[placenum]+'_observed')
            ax.set_xlabel('ECM percent (%)', fontsize = 8)
            plt.xticks(fontsize=8)
            ax.set_ylabel('Unprotected SOC (% soil mass)', fontsize = 8)
            plt.yticks(fontsize=8)
            plt.legend(frameon=False, prop={"size": 8})
plt.figure('Scatterplot of unprotected SOC', figsize=(6, 6));
plt.scatter(obs_UnpC, plot_UnpC[:, 0, 0], s=120, marker='o', facecolors='none', edgecolors='b', alpha = 0.7)
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.xlabel('Observed unprotected SOC (mg/g soil)')
plt.ylabel('Simulated unprotected SOC (mg/g soil)')
z = numpy.polyfit(obs_UnpC, plot_UnpC[:, 0, 0], 1)
p = numpy.poly1d(z)
plt.plot(obs_UnpC, p(obs_UnpC), "r-")
stats_results = polyfit(obs_UnpC, plot_UnpC[:, 0, 0], 1)
r_squared = stats_results['determination']
plt.text(4, 6, '$R^2$ = {R_squared:2.3f}'.format(R_squared=r_squared), fontsize=15)

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

plt.figure('Monthly SAP N sources from Necromass', figsize=(6, 8));
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

plt.figure('Monthly SAP N limit status', figsize=(6, 8));
plt.clf()
time = numpy.arange(0, 1+timestep, timestep)
plot_Nlimit_Monthly = numpy.zeros(1+int(1/timestep))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_Nlimit_Monthly[i + 1+int(1/timestep)] = 100*(1-plot_Nlimit_all[i, plotnum, claynum, climnum])
            plt.plot(time, plot_Nlimit_Monthly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            if climnum == 0:
                plt.ylabel('SAP N limit status (%)')

plt.figure('Monthly AM Intermediate N', figsize=(6, 8));
plt.clf()
time = numpy.arange(0, 1+timestep, timestep)
plot_IntN_AM_Monthly = numpy.zeros(1+int(1/timestep))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_IntN_AM_Monthly[i + 1+int(1/timestep)] = plot_IntN_AM_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_IntN_AM_Monthly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')
            if climnum == 0:
                plt.ylabel('Monthly Intermediate N')

plt.figure('Monthly ECM Intermediate N', figsize=(6, 8));
plt.clf()
time = numpy.arange(0, 1+timestep, timestep)
plot_IntN_ECM_Monthly = numpy.zeros(1+int(1/timestep))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_IntN_ECM_Monthly[i + 1+int(1/timestep)] = plot_IntN_ECM_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_IntN_ECM_Monthly[:], ms=4, marker=markers[claynum],
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
# ax.set_title("Monthly total MYC transfer")
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
# ax.set_title("Monthly ECM transfer")
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
# ax.set_title("Monthly AM transfer")
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
# ax.set_title("Monthly Root uptake")
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
ax = plt.subplot(231)
# ax.set_title("Annual mycorrhizal transfer")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        plt.plot(ECM_pct, plot_Ntransfer[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        # plt.xlabel('ECM percent (%)')
        plt.ylabel('Annual mycorrhizal transfer')
ax = plt.subplot(232)
# ax.set_title("Annual ECM transfer")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        plt.plot(ECM_pct, plot_Ntransfer_ECM[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        # plt.xlabel('ECM percent (%)')
        plt.ylabel('Annual ECM transfer')
ax = plt.subplot(233)
# ax.set_title("Annual AM transfer")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        # print(climnum, plot_Ntransfer_AM[:, claynum, climnum])
        plt.plot(ECM_pct, plot_Ntransfer_AM[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        # plt.xlabel('ECM percent (%)')
        plt.ylabel('Annual AM transfer')
ax = plt.subplot(234)
# ax.set_title("Root N uptake")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        # print(climnum, plot_Nrootuptake[:, claynum, climnum])
        plt.plot(ECM_pct, plot_Nrootuptake[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        plt.xlabel('ECM percent (%)')
        plt.ylabel('Root N uptake')
ax = plt.subplot(235)
ax.set_title("Total N uptake")
for claynum in range(len(clay)):
    for climnum in range(len(MAT)):
        # print(climnum,plot_Nrootuptake[:, claynum, climnum]+plot_Ntransfer[:, claynum, climnum])
        plt.plot(ECM_pct, plot_Nrootuptake[:, claynum, climnum]+plot_Ntransfer[:, claynum, climnum], ms=4, marker=markers[claynum],
                 c=cmap(norm(MAT[climnum])),
                 label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
        plt.xlabel('ECM percent (%)')
        plt.ylabel('Total N uptake')

plt.figure('falloc', figsize=(6, 8));
plt.clf()
time = numpy.arange(0, 1+timestep, timestep)
plot_falloc_ECM_Monthly = numpy.zeros(1+int(1/timestep))
plot_falloc_AM_Monthly = numpy.zeros(1+int(1/timestep))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_falloc_ECM_Monthly[i + 1+int(1/timestep)] = 100*plot_falloc_ECM_all[i, plotnum, claynum, climnum]
                plot_falloc_AM_Monthly[i + 1 + int(1 / timestep)] = 100 * plot_falloc_AM_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_falloc_ECM_Monthly[:], ms=4, marker='o',
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.plot(time, plot_falloc_AM_Monthly[:], ms=4, marker='s',
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')

plt.figure('Int_ECMC', figsize=(6, 8));
plt.clf()
time = numpy.arange(0, 1+timestep, timestep)
plot_IntECMC_Monthly = numpy.zeros(1+int(1/timestep))
for plotnum in range(len(ECM_pct)):
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            for i in numpy.arange(-(1+int(1/timestep)), 0, 1):
                plot_IntECMC_Monthly[i + 1+int(1/timestep)] = plot_IntECMC_all[i, plotnum, claynum, climnum]
            plt.plot(time, plot_IntECMC_Monthly[:], ms=4, marker=markers[claynum],
                     c=cmapECM(normECM(ECM_pct[plotnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('Time (year)')

# plt.figure('falloc_barplot', figsize=(6, 8));
# plt.clf()
# time = numpy.arange(0, 1+timestep, timestep)
# for claynum in range(len(clay)):
#     for climnum in range(len(MAT)):
#         ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
#         ax.set_title(str(MAT[climnum])+"°C")
#         x = numpy.arange(5)
#         width = 0.25
#         plt.bar(x - 0.5*width, plot_falloc_AM[:, claynum, climnum], width, color='cyan')
#         plt.bar(x + 0.5*width, plot_falloc_ECM[:, claynum, climnum], width, color='orange')
#         plt.xticks(x, ['0%', '25%', '50%', '75%', '100%'])
#         plt.xlabel("ECM gradient")
#         plt.ylabel("falloc")
#         plt.legend(["NPP% to AM", "NPP% to ECM"])
#
# plt.figure('N uptake_barplot', figsize=(6, 8));
# plt.clf()
# time = numpy.arange(0, 1+timestep, timestep)
# for claynum in range(len(clay)):
#     for climnum in range(len(MAT)):
#         ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
#         ax.set_title(str(MAT[climnum])+"°C")
#         x = numpy.arange(5)
#         width = 0.2
#         plt.bar(x - width, plot_Ntransfer_AM[:, claynum, climnum], width, color='cyan')
#         plt.bar(x, plot_Ntransfer_ECM[:, claynum, climnum], width, color='orange')
#         plt.bar(x + width, plot_Nrootuptake[:, claynum, climnum], width, color='green')
#         plt.xticks(x, ['0%', '25%', '50%', '75%', '100%'])
#         plt.xlabel("ECM gradient")
#         plt.ylabel("N uptake")
#         plt.legend(["AM transfer", "ECM transfer", "Root uptake"])

plt.show()