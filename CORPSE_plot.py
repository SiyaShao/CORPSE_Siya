# Plot the results
import pandas, numpy
import matplotlib.pyplot as plt

def plot(timestep, timesteps, nplots, nclays, nclimates, ECM_pct, clay, MAT):

    norm = plt.Normalize(5, 20)
    normECM = plt.Normalize(0, 100)
    cmap = plt.get_cmap('rainbow')
    cmapECM = plt.get_cmap('summer')
    markers = ['o', 's']
    SMALL_SIZE = 8

    # Construct the datasets for plotting
    # Store the data along all the time steps
    data = numpy.zeros([timesteps, nplots, nclays, nclimates])
    plot_InorgN_all, plot_SAPC_all, plot_UnpC_all, plot_ECMC_all, plot_AMC_all, plot_TotN_all, plot_NfromNecro_all, \
    plot_NfromSOM_all, plot_Nsource_all, plot_Nlimit_all, plot_Nlimit_acc = data.copy(), data.copy(), data.copy(), \
                                                                            data.copy(), data.copy(), data.copy(), \
                                                                            data.copy(), data.copy(), data.copy(), \
                                                                            data.copy(), data.copy()
    plot_IntN_ECM_all, plot_Nrootuptake_all, plot_Ntransfer_all, plot_Ntransfer_ECM_all, plot_Ntransfer_AM_all, \
    plot_Nrootuptake_acc, plot_Ntransfer_acc, plot_Ntransfer_ECM_acc, plot_Ntransfer_AM_acc, \
    plot_falloc_ECM_all, plot_falloc_ECM_acc = data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), \
                                       data.copy(), data.copy(), data.copy(), data.copy(), data.copy()
    plot_IntECMC_all, plot_TotC_all, plot_falloc_AM_all, plot_falloc_AM_acc = data.copy(), data.copy(), data.copy(), data.copy()
    plot_SumderivsC_acc, plot_SumderivsN_acc, plot_SumderivsC_all, plot_SumderivsN_all = data.copy(), data.copy(),\
                                                                                         data.copy(), data.copy()

    # Store the annual data of the last year
    data = numpy.zeros([nplots, nclays, nclimates])
    plot_InorgN, plot_SAPC, plot_UnpC, plot_ECMC, plot_AMC, plot_TotN, plot_NfromNecro, plot_NfromSOM, plot_Nsource, \
    plot_Nlimit, plot_TotC = data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), data.copy(), \
                             data.copy(), data.copy(), data.copy(), data.copy()
    plot_Nrootuptake, plot_Ntransfer, plot_Ntransfer_ECM, plot_Ntransfer_AM, plot_falloc_ECM = data.copy(), data.copy(), \
                                                                                           data.copy(), data.copy(), data.copy()
    plot_falloc_AM, plot_falloc_ECM, plot_SumderivsC, plot_SumderivsN = data.copy(),data.copy(),data.copy(),data.copy()

    for plotnum in range(nplots):
        for claynum in range(nclays):
            for climnum in range(nclimates):
                filename = "/Users/f0068s6/PycharmProjects/CORPSE_Siya/" + str(nclimates * nclays * plotnum + nclimates * claynum + climnum + 1)\
                           + '_Monthly_data.txt'
                result = pandas.read_csv(filename)
                plot_InorgN_all[:, plotnum, claynum, climnum] = result['inorganicN']
                plot_SAPC_all[:, plotnum, claynum, climnum] = result['SAPC']
                plot_UnpC_all[:, plotnum, claynum, climnum] = result['uFastC']+result['uSlowC']+result['uNecroC']
                plot_TotC_all[:, plotnum, claynum, climnum] = plot_UnpC_all[:, plotnum, claynum, climnum]+\
                                                              result['pFastC']+result['pSlowC']+result['pNecroC']
                plot_ECMC_all[:, plotnum, claynum, climnum] = result['ECMC']
                plot_AMC_all[:, plotnum, claynum, climnum] = result['AMC']
                plot_TotN_all[:, plotnum, claynum, climnum] = result['uFastN']+result['uSlowN']+result['uNecroN']+\
                                                              result['pFastN']+result['pSlowN']+result['pNecroN']
                plot_NfromNecro_all[:, plotnum, claynum, climnum] = result['NfromNecro']
                plot_NfromSOM_all[:, plotnum, claynum, climnum] = result['NfromSOM']
                plot_Nlimit_acc[:, plotnum, claynum, climnum] = result['Nlimit']
                plot_IntN_ECM_all[:, plotnum, claynum, climnum] = result['Int_N_ECM']
                plot_Ntransfer_acc[:, plotnum, claynum, climnum] = result['Ntransfer']
                plot_Ntransfer_ECM_acc[:, plotnum, claynum, climnum] = result['Ntransfer_ECM']
                plot_Ntransfer_AM_acc[:, plotnum, claynum, climnum] = result['Ntransfer_AM']
                plot_Nrootuptake_acc[:, plotnum, claynum, climnum] = result['Nrootuptake']
                plot_falloc_ECM_acc[:, plotnum, claynum, climnum] = result['falloc_ECM']
                plot_falloc_AM_acc[:, plotnum, claynum, climnum] = result['falloc_AM']
                plot_IntECMC_all[:, plotnum, claynum, climnum] = result['Int_ECMC']

                plot_SumderivsC_acc[:, plotnum, claynum, climnum] = result['Sum_derivs_C']
                plot_SumderivsN_acc[:, plotnum, claynum, climnum] = result['Sum_derivs_N']

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
                        plot_SumderivsC_all[i, plotnum, claynum, climnum] = plot_SumderivsC_acc[i, plotnum, claynum, climnum]
                        plot_SumderivsN_all[i, plotnum, claynum, climnum] = plot_SumderivsN_acc[i, plotnum, claynum, climnum]
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
                        plot_SumderivsC_all[i, plotnum, claynum, climnum] = 1/timestep * \
                             (plot_SumderivsC_acc[i, plotnum, claynum, climnum]-plot_SumderivsC_acc[i-1, plotnum, claynum, climnum])
                        plot_SumderivsN_all[i, plotnum, claynum, climnum] = 1/timestep * \
                             (plot_SumderivsN_acc[i, plotnum, claynum, climnum]-plot_SumderivsN_acc[i-1, plotnum, claynum, climnum])

                average_time = 10*int(1/timestep)
                factor_time = 1/average_time
                for a in range(average_time):
                    plot_InorgN[plotnum, claynum, climnum] += factor_time * (plot_InorgN_all[-average_time + a, plotnum, claynum, climnum])
                    plot_SAPC[plotnum, claynum, climnum] += factor_time * (plot_SAPC_all[-average_time + a, plotnum, claynum, climnum])
                    plot_UnpC[plotnum, claynum, climnum] += factor_time * (plot_UnpC_all[-average_time + a, plotnum, claynum, climnum])
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
                    plot_SumderivsC[plotnum, claynum, climnum] += factor_time * (plot_SumderivsC_all[-average_time+ a, plotnum, claynum, climnum])
                    plot_SumderivsN[plotnum, claynum, climnum] += factor_time * (plot_SumderivsN_all[-average_time+ a, plotnum, claynum, climnum])

    plt.figure('Total soil C:N', figsize=(6, 8));
    plt.clf()
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            plt.plot(ECM_pct, plot_TotC[:, claynum, climnum]/plot_TotN[:, claynum, climnum], ms=4, marker=markers[claynum],
                     c=cmap(norm(MAT[climnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('ECM percent (%)')

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
        # plt.ylim([0, 0.001])

    plt.figure('Long-term unprotected C under 5 degree 10%Clay', figsize=(6, 8));
    plt.clf()
    for plotnum in range(len(ECM_pct)):
        plt.plot(plottimes,plot_UnpC_all[:,plotnum,0,0],c=cmapECM(normECM(ECM_pct[plotnum])))
        plt.xlabel('Time (year)')
        plt.ylabel('Unprotected C (KgC/m2)')
        # plt.ylim([0, 10.0])

    plt.figure('Long-term total N under 5 degree 10%Clay', figsize=(6, 8));
    plt.clf()
    for plotnum in range(len(ECM_pct)):
        plt.plot(plottimes,plot_TotN_all[:,plotnum,0,0],c=cmapECM(normECM(ECM_pct[plotnum])))
        plt.xlabel('Time (year)')
        plt.ylabel('Total N (KgN/m2)')
        # plt.ylim([0.2, 1.2])

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

    plt.figure('Monthly Intermediate N', figsize=(6, 8));
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
            plt.plot(ECM_pct, plot_Ntransfer_AM[:, claynum, climnum], ms=4, marker=markers[claynum],
                     c=cmap(norm(MAT[climnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            # plt.xlabel('ECM percent (%)')
            plt.ylabel('Annual AM transfer')
    ax = plt.subplot(234)
    # ax.set_title("Root N uptake")
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            plt.plot(ECM_pct, plot_Nrootuptake[:, claynum, climnum], ms=4, marker=markers[claynum],
                     c=cmap(norm(MAT[climnum])),
                     label='Clay={claypct:1.1f}%, MAT={mat:1.1f}C'.format(claypct=clay[claynum], mat=MAT[climnum]))
            plt.xlabel('ECM percent (%)')
            plt.ylabel('Root N uptake')
    ax = plt.subplot(235)
    ax.set_title("Total N uptake")
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
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

    plt.figure('falloc_barplot', figsize=(6, 8));
    plt.clf()
    time = numpy.arange(0, 1+timestep, timestep)
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            x = numpy.arange(5)
            width = 0.25
            plt.bar(x - 0.5*width, plot_falloc_AM[:, claynum, climnum], width, color='cyan')
            plt.bar(x + 0.5*width, plot_falloc_ECM[:, claynum, climnum], width, color='orange')
            plt.xticks(x, ['0%', '25%', '50%', '75%', '100%'])
            plt.xlabel("ECM gradient")
            plt.ylabel("falloc")
            plt.legend(["NPP% to AM", "NPP% to ECM"])

    plt.figure('Derivs check', figsize=(6, 8));
    plt.clf()
    time = numpy.arange(0, 1+timestep, timestep)
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            x = numpy.arange(5)
            width = 0.25
            # plt.bar(x - 0.5*width, plot_SumderivsC[:, claynum, climnum], width, color='cyan')
            plt.bar(x + 0.5*width, plot_SumderivsN[:, claynum, climnum], width, color='orange')
            plt.xticks(x, ['0%', '25%', '50%', '75%', '100%'])
            plt.xlabel("ECM gradient")
            plt.ylabel("falloc")
            plt.legend(["NPP% to AM", "NPP% to ECM"])

    plt.figure('N uptake_barplot', figsize=(6, 8));
    plt.clf()
    time = numpy.arange(0, 1+timestep, timestep)
    for claynum in range(len(clay)):
        for climnum in range(len(MAT)):
            ax = plt.subplot(int("1"+str(nclimates)+str(climnum+1)))
            ax.set_title(str(MAT[climnum])+"°C")
            x = numpy.arange(5)
            width = 0.2
            plt.bar(x - width, plot_Ntransfer_AM[:, claynum, climnum], width, color='cyan')
            plt.bar(x, plot_Ntransfer_ECM[:, claynum, climnum], width, color='orange')
            plt.bar(x + width, plot_Nrootuptake[:, claynum, climnum], width, color='green')
            plt.xticks(x, ['0%', '25%', '50%', '75%', '100%'])
            plt.xlabel("ECM gradient")
            plt.ylabel("N uptake")
            plt.legend(["AM transfer", "ECM transfer", "Root uptake"])

    plt.show()

# Plot the results
timestep = 1/12.0  # Monthly
finaltimes = numpy.arange(0, 100 + timestep, timestep)  # Time steps to evaluat, running on quatily timestep
plottimes = finaltimes
timesteps = len(finaltimes)

nplots = 5
nclays = 1
nclimates = 3
ECM_pct = numpy.linspace(0, 100, nplots)
MAT = numpy.linspace(5, 20, nclimates)
clay = numpy.linspace(10, 70, nclays)

# plot(timestep, timesteps, nplots, nclays, nclimates, ECM_pct, clay, MAT)