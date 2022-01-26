#This is for the new branch "Add-new-mycorrhizal-types"

expected_params={	'vmaxref': 'Relative maximum enzymatic decomp rates (length 3)',
        	'Ea':	'Activation energy (length 3)',
        	'kC':	'Michaelis-Menton parameter (length 3)',
        	'gas_diffusion_exp': 'Determines suppression of decomp at high soil moisture',
            'substrate_diffusion_exp':'Determines suppression of decomp at low soil moisture',
        	'minMicrobeC':	   'Minimum microbial biomass (fraction of total C)',
        	'Tmic': 'Microbial lifetime at 20C (years) (length 3)',
        	'et': 'Fraction of turnover not converted to CO2 (length 3)',
        	'eup': 'Carbon uptake efficiency (length 3)',
            'nup': 'Nitrogen uptake efficiency (length 3)',
        	'tProtected':	'Protected C turnover time (years)',
        	'protection_rate':'Protected carbon formation rate (year-1) (length 3)',
            'CN_microbe': 'C:N ratio of microbial biomass (length 3)',
            'frac_N_turnover_min': 'Fraction of microbial biomass N turnover that is mineralized',
            'frac_turnover_slow': 'Fraction of microbial biomass N turnover that goes to slow pool (length 3)',
            'max_immobilization_rate': 'Maximum N immobilization rate (fraction per day) (length 3)',
            'new_resp_units':'If true, vmaxref has units of 1/years and assumes optimal soil moisture has a relative rate of 1.0',
            'eup_myc':'Carbon uptake efficiency for mycorrhizal fungi (length 2)',
            'max_mining_rate': 'Maximum N mining rates from unprotected soil organic N pools (ECM fungi) (length 3)',
            'kc_mining': 'Half-saturation constant of ECM biomass concentration for ECM N mining',
            'max_scavenging_rate':'Maximum N scavenging rates from soil inorganic N pools (AM fungi)',
            'kc_scavenging': 'Half-saturation constant of AM biomass concentration for AM N scavenging',
            'kc_scavenging_IN': 'Half-saturation constant of inorganic N concentration for AM N scavenging',
            'Ea_inorgN': 'Activation energy for immobilization of inorganic N',
            'depth': 'Soil depth'
            }

chem_types = ['Fast','Slow','Necro']

# This sets up three types of microbes: one is free-living saprotrophs, the other two are ECM and AM mycorrhizal fungi
mic_types = ['SAP','ECM','AM']

# This sets up pools like "uFastC" for unprotected fast C and "pNecroN" for protected necromass N
expected_pools = ['u'+t+'C' for t in chem_types]+\
                 ['p'+t+'C' for t in chem_types]+\
                 ['u'+t+'N' for t in chem_types]+\
                 ['p'+t+'N' for t in chem_types]+\
                 [mt+'C' for mt in mic_types]   +\
                 [mt+'N' for mt in mic_types]   +\
                 ['CO2','inorganicN']           +\
                 ['Int_ECMC', 'Int_AMC']
#                 ['livingMicrobeC','livingMicrobeN','CO2','inorganicN',]


#All soils: slope=0.4833,intercept=2.3282
#Alfisols: slope=0.5945, intercept=2.2788
def prot_clay(claypercent,slope=0.4833,intercept=2.3282,BD=1.15,porosity=0.4):
    ''' Calculate protection rate as a function of clay content, based on sorption isotherms from Mayes et al (2012) Table 3
    Calculates Qmax in mgC/kg soil from Mayes et al 2012, converted to g/m3 using bulk density
    Typically used as relative value for calculating protection_rate parameter.
    claypercent: Soil % clay (out of 100)
    slope: Either for all soils or a soil order, from Mayes paper
    intercept: Either for all soils or a soil order, from Mayes paper
    BD: Soil bulk density in g/cm3
    '''
    from numpy import log10,where,atleast_1d
    prot=where(atleast_1d(claypercent)!=0.0,1.0*(10**(slope*log10(claypercent)+intercept)*BD*1e-6),0.0)
    return prot


def check_params(params):
    '''params: dictionary containing parameter values. Must contain all the fields in expected_params'''

    from numpy import iterable,array
    unused_params=expected_params.copy()
    for k in params.keys():
        if k not in expected_params:
            raise ValueError('Parameter set contains unexpected parameter %s'%k)
        unused_params.pop(k)
        if iterable(params[k]):
            params[k]=array(params[k])
    if len(unused_params)>0:
        for k in unused_params.keys():
            print ('Missing parameter: %s [%s]'%(k,unused_params[k]))
        raise ValueError('Missing parameters: %s'%unused_params.keys())


from numpy import zeros,size,where,atleast_1d,zeros_like
def CORPSE_deriv(SOM,T,theta,Ndemand,params,claymod=1.0):
    '''Calculate rates of change for all CORPSE pools
       T: Temperature (K)
       theta: Soil water content (fraction of saturation)

       Returns same data structure as SOM'''

    # if any(T<0.0):
    #     raise ValueError('T must be >=0')

    theta=atleast_1d(theta)
    T=atleast_1d(T)

    theta[theta<0]=0.0
    theta[theta>1]=1.0

    et=params['et']
    eup=params['eup']

    # Calculate maximum potential C decomposition rate
    decomp=decompRate(SOM,T,theta,params)

    # Microbial turnover
    microbeTurnover = {'SAP':0.0,'ECM':0.0,'AM':0.0}
    maintenance_resp = {'SAP':0.0,'ECM':0.0,'AM':0.0}
    overflow_resp = {'SAP':0.0,'ECM':0.0,'AM':0.0}
    carbon_supply = {'SAP':0.0,'ECM':0.0,'AM':0.0}
    nitrogen_supply = {'SAP':0.0,'ECM':0.0,'AM':0.0}
    deadmic_C_production = {'SAP':0.0,'ECM':0.0,'AM':0.0}
    deadmic_N_production = {'SAP':0.0,'ECM':0.0,'AM':0.0}
    dmicrobeC = {'SAP':0.0,'ECM':0.0,'AM':0.0}
    dmicrobeN = {'SAP':0.0,'ECM':0.0,'AM':0.0}
    CN_imbalance_term = {'SAP':0.0,'ECM':0.0,'AM':0.0}
    # Separate microbial metabolisms among different microbial types

    Nacq_simb_max = {'ECM':0.0,'AM':0.0}
    Nmining = NminingRate(SOM,T,theta,params)
    Nacq_simb_max['ECM'] = sum(Nmining.values())
    Nacq_simb_max['AM'] = T_factor(T,params) * params['max_scavenging_rate']['AM'] * SOM['inorganicN'] / (
            SOM['inorganicN'] + params['kc_scavenging_IN']['AM'] * params['depth']) \
                          * SOM['AMC'] / (SOM['AMC'] + params['kc_scavenging']['AM'] * params['depth'])
    # Calculate potential ECM N mining and AM N scavenging

    for mt in mic_types:
        microbeTurnover[mt]=(SOM[mt+'C']-params['minMicrobeC']*(sumCtypes(SOM,'u')))/params['Tmic'][mt];   # kg/m2/yr
        if isinstance(microbeTurnover[mt],float):
           microbeTurnover[mt]=max(0.0,microbeTurnover[mt])
        else:
           microbeTurnover[mt][microbeTurnover[mt]<0.0]=0.0

        maintenance_resp[mt]=microbeTurnover[mt]*(1.0-et[mt])

        deadmic_C_production[mt]=microbeTurnover[mt]*et[mt]   # actual fraction of microbial turnover
        deadmic_N_production[mt]=microbeTurnover[mt]/params['CN_microbe'][mt]
        # Note that we haven't set up mycorrhizal N cycle yet thus we only consider SAP CN interaction for now.

        # C and N available for microbial growth
        if mt=='SAP':
           for t in chem_types:
               carbon_supply[mt]=carbon_supply[mt]+decomp[t+'C']*params['eup'][t]
               nitrogen_supply[mt]=nitrogen_supply[mt]+decomp[t+'N']*params['nup'][t]
               IMM_N_max = T_factor(T,params) * params['max_scavenging_rate']['SAP'] * SOM['inorganicN'] / (
                       SOM['inorganicN'] + params['kc_scavenging_IN']['SAP'] * params['depth']) \
                                 * SOM['SAPC'] / (SOM['SAPC'] + params['kc_scavenging']['SAP'] * params['depth'])
        else:
           carbon_supply[mt] += SOM['Int_'+mt+'C'] * params['eup_myc'][mt]
           nitrogen_supply[mt] += Nacq_simb_max[mt]
           maintenance_resp[mt] += SOM['Int_' + mt + 'C'] * (1-params['eup_myc'][mt])
           IMM_N_max=atleast_1d(0.0)

        # Growth is nitrogen limited, with not enough mineral N to support it with max immobilization
        # loc_Nlim is originally a vector of True/False that tells the code where this condition applies
        # Now change loc_Nlim, loc_immob and loc_Clim to values 0/1 to make it more concise
        loc_Nlim = int((carbon_supply[mt])>((nitrogen_supply[mt]+IMM_N_max)*params['CN_microbe'][mt]))
        if loc_Nlim==1:
            if mt == 'SAP':
                print('SAPs are N limiting')
            CN_imbalance_term[mt] = -IMM_N_max*loc_Nlim
            dmicrobeC[mt] = ((nitrogen_supply[mt]+IMM_N_max)*params['CN_microbe'][mt] - microbeTurnover[mt])*loc_Nlim
            dmicrobeN[mt] = dmicrobeC[mt]/params['CN_microbe'][mt]
            overflow_resp[mt] = (carbon_supply[mt]-(nitrogen_supply[mt]+IMM_N_max)*params['CN_microbe'][mt])*loc_Nlim
        else:
            # Growth is ultimately carbon limited
            # but must be supported by immobilization of some mineral nitrogen or extra N is mineralized
            # if mt == 'SAP':
            #     loc_immob = int(carbon_supply[mt] >= nitrogen_supply[mt] * params['CN_microbe'][mt]) & (
            #                 carbon_supply[mt] < (nitrogen_supply[mt] + IMM_N_max) * params['CN_microbe'][mt])
            # For MYC fungi, since IMM_N_max = 0, this refers to the situation where extra N is transported to plants
            loc_Clim = 1
            CN_imbalance_term[mt] = (nitrogen_supply[mt] - carbon_supply[mt] / params['CN_microbe'][mt]) * loc_Clim
            dmicrobeC[mt] = (carbon_supply[mt] - microbeTurnover[mt]) * loc_Clim
            dmicrobeN[mt] = dmicrobeC[mt] / params['CN_microbe'][mt]

    # If mycorrhizal fungi transfer too much N to plants (exceeding plant N demands), then mycorrhizal N acquisition
    # is decreased accordingly. This will not be needed once coupled to a plant growth model.
    Ntransfer = max(0.0,CN_imbalance_term['ECM']) + max(0.0,CN_imbalance_term['AM'])
    if Ntransfer>Ndemand:
       Nmining_d = (Ntransfer-Ndemand)*max(0.0,CN_imbalance_term['ECM'])/Ntransfer
       for t in chem_types:
           if nitrogen_supply['ECM']>0.0:
              Nmining[t+'N'] = Nmining[t+'N']*(nitrogen_supply['ECM']-Nmining_d)/nitrogen_supply['ECM']
       Nscavenging_d = Ntransfer-Ndemand-Nmining_d
       nitrogen_supply['AM'] += -Nscavenging_d
    else:
       print('!! Unbalanced N budget for the plants')

    # CO2 production and cumulative CO2 produced by cohort
    CO2prod = sum(maintenance_resp.values()) + sum(overflow_resp.values()) # Sum up all the CO2 production from different microbial groups
    for t in chem_types:
        CO2prod=CO2prod+decomp[t+'C']*(1.0-eup[t])


    # Update protected carbon
    protectedCturnover = dict([(t,SOM['p'+t+'C']/params['tProtected']) for t in chem_types])
    protectedCprod =     dict([(t,SOM['u'+t+'C']*params['protection_rate'][t]*claymod) for t in chem_types])
    protectedNturnover = dict([(t,SOM['p'+t+'N']/params['tProtected']) for t in chem_types])
    protectedNprod =     dict([(t,SOM['u'+t+'N']*params['protection_rate'][t]*claymod) for t in chem_types])

    derivs=SOM.copy()
    for k in derivs.keys():
        derivs[k]=0.0

    derivs['SAPC']=atleast_1d(dmicrobeC['SAP'])
    derivs['SAPN']=atleast_1d(dmicrobeN['SAP']) # Will change to "for mt in mic_types" later on for mycorrhizal fungi
    derivs['CO2'] =atleast_1d(CO2prod)

    derivs['inorganicN'] += CN_imbalance_term['SAP']-nitrogen_supply['AM']-SOM['inorganicN']*params['iN_loss_rate'] + \
        params['N_deposition']  # SAP net N mineralization + AM N scavenging - N loss + N deposition

    for t in chem_types:
        derivs['inorganicN'] += decomp[t+'N']*(1-params['nup'][t])

    for t in chem_types:
        derivs['u'+t+'C']=-decomp[t+'C']+protectedCturnover[t]-protectedCprod[t]
        derivs['p'+t+'C']=protectedCprod[t]-protectedCturnover[t]
        derivs['u'+t+'N']=-decomp[t+'N']+protectedNturnover[t]-protectedNprod[t]-Nmining[t+'N']
        # N loss in decomposition & N transferred to protected pools & N mining from ECM
        derivs['p'+t+'N']=protectedNprod[t]-protectedNturnover[t]

    for mt in mic_types:
        derivs['uNecroC'] += deadmic_C_production[mt]*(1.0-params['frac_turnover_slow'][mt])
        derivs['uSlowC']  += deadmic_C_production[mt]*params['frac_turnover_slow'][mt]
        turnover_N_min    =  deadmic_N_production[mt]*params['frac_N_turnover_min'];
        turnover_N_slow   =  deadmic_N_production[mt]*(1.0-params['frac_N_turnover_min'])*params['frac_turnover_slow'][mt];
        derivs['uNecroN'] += deadmic_N_production[mt]-turnover_N_min-turnover_N_slow
        derivs['uSlowN']  += turnover_N_slow
        derivs['inorganicN']  += turnover_N_min

    derivs['ECMC'] = atleast_1d(dmicrobeC['ECM'])
    derivs['ECMN'] = atleast_1d(dmicrobeN['ECM'])
    derivs['AMC'] = atleast_1d(dmicrobeC['AM'])
    derivs['AMN'] = atleast_1d(dmicrobeN['AM'])

    derivs['Int_ECMC'] = atleast_1d(-SOM['Int_ECMC'])
    derivs['Int_AMC'] = atleast_1d(-SOM['Int_AMC'])

    return derivs


# Decomposition rate
def decompRate(SOM,T,theta,params):

    # This only really needs to be calculated once
    if params['new_resp_units']:
        theta_resp_max=params['substrate_diffusion_exp']/(params['gas_diffusion_exp']*(1.0+params['substrate_diffusion_exp']/params['gas_diffusion_exp']))
        aerobic_max=theta_resp_max**params['substrate_diffusion_exp']*(1.0-theta_resp_max)**params['gas_diffusion_exp']
    else:
        aerobic_max=1.0

    vmax=Vmax(T,params,'decompo')

    decompRate={}
    dodecomp=atleast_1d((sumCtypes(SOM,'u')!=0.0)&(theta!=0.0)&(SOM['SAPC']!=0.0))
    for t in chem_types:
        if dodecomp.any():
            drate=where(dodecomp,vmax[t]*theta**params['substrate_diffusion_exp']*(SOM['u'+t+'C'])*SOM['SAPC']/(sumCtypes(SOM,'u')*params['kC'][t]+SOM['SAPC'])*(1.0-theta)**params['gas_diffusion_exp']/aerobic_max,0.0)
        decompRate[t+'C']=drate
        decompRate[t+'N']=where(SOM['u'+t+'C']>0,drate*SOM['u'+t+'N']/SOM['u'+t+'C'],0.0)

    return decompRate

# N mining rate
def NminingRate(SOM,T,theta,params):

    # This only really needs to be calculated once
    if params['new_resp_units']:
        theta_resp_max=params['substrate_diffusion_exp']/(params['gas_diffusion_exp']*(1.0+params['substrate_diffusion_exp']/params['gas_diffusion_exp']))
        aerobic_max=theta_resp_max**params['substrate_diffusion_exp']*(1.0-theta_resp_max)**params['gas_diffusion_exp']
    else:
        aerobic_max=1.0

    vmax=Vmax(T,params,'Nmining')

    NminingRate={}
    dodecomp=atleast_1d((sumCtypes(SOM,'u')!=0.0)&(theta!=0.0)&(SOM['SAPC']!=0.0))
    for t in chem_types:
        if dodecomp.any():
           drate=where(dodecomp,vmax[t]*theta**params['substrate_diffusion_exp']*(SOM['u'+t+'C'])*SOM['ECMC']/(sumCtypes(SOM,'u')*params['kc_mining']+SOM['ECMC'])*(1.0-theta)**params['gas_diffusion_exp']/aerobic_max,0.0)
        NminingRate[t+'N']=where(SOM['u'+t+'C']>0,drate*SOM['u'+t+'N']/SOM['u'+t+'C'],0.0)

    return NminingRate

def Vmax(T,params,process):
    '''Vmax function, normalized to Tref=293.15
    T is in K'''

    Tref=293.15;
    Rugas=8.314472;

    from numpy import exp

    if process=='decompo':
       Vmax=dict([(t,params['vmaxref'][t]*exp(-params['Ea'][t]*(1.0/(Rugas*T)-1.0/(Rugas*Tref)))) for t in chem_types]);
    elif process=='Nmining':
       Vmax=dict([(t,params['max_mining_rate'][t]*exp(-params['Ea'][t]*(1.0/(Rugas*T)-1.0/(Rugas*Tref)))) for t in chem_types])

    return Vmax

def T_factor(T,params):

    Tref=293.15;
    Rugas=8.314472;

    from numpy import exp

    T_factor = exp(-params['Ea_inorgN'] * (1.0 / (Rugas * T) - 1.0 / (Rugas * Tref)))

    return T_factor

def sumCtypes(SOM,prefix,suffix='C'):
    out=SOM[prefix+chem_types[0]+suffix]
    if len(chem_types)>1:
        for t in chem_types[1:]:
            out=out+SOM[prefix+t+suffix]

    return out
