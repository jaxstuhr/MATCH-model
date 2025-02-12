# Copyright (c) 2022 The MATCH Authors. All rights reserved.
# Licensed under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3 (or later), which is in the LICENSE file.

"""
Determines the resource adequacy value for built resources and adds RA open position cost to the objective function

Starting in 2025, the CPUC dramatically changed the Resource Adequacy requirements that LSE's ust meet to meet compliance.
The new requirement is called "Slice of Day". Each LSE has a 24-hour profile of its Peak Demand (MW) for each month that it must meet from RA eligible resources.
Variable Gen Resources contribute to RA based on an exceedance values determined by the CPUC based on historical generation during peak demand hours.
Hydro resources contribute to RA based on a monthly NQC values. 
Storage resources allow LSEs to shift RA between hours but have no qualifying capacity themselves. 
A detailed explanation of the SOD framework can be found here: https://www.cpuc.ca.gov/-/media/cpuc-website/divisions/energy-division/documents/resource-adequacy-homepage/resource-adequacy-compliance-materials/guides-and-resources/2025-ra-slice-of-day-filing-guide.pdf

This module attributes RA value to resources in a portfolio and optimizes the RA position to minimize required procurement of 24/7 RA products. The cost of this new RA is added to the objective function.
"""

import os
from pyomo.environ import *
import pandas as pd

dependencies = (
    "match_model.timescales",
    "match_model.financials",
    "match_model.balancing.load_zones",
    "match_model.generators.build",
    "match_model.generators.dispatch",
    "match_model.generators.storage",
)


def define_arguments(argparser):
    argparser.add_argument(
        "--sell_excess_RA",
        choices=["none", "sell"],
        default="none",
        help="Whether or not to consider sold excess RA in the objective function. "
        "Specify 'none' to disable.",
    )


def define_components(mod):
    
    ## For each month, we calculate NOP[t] = RA_Obl[t] * PRM - PPA_RA[t] - Contract_RA[t] - Alloc_RA[t] for t in 1:24
    ## For each month there is a Storage_P and Storage_E input that are based on our portfolio
    ## For all months, there is a storage efficiency input Storage_RTE
    ## There are 4 variables to be optimized:
    ## 1. NewRA_P which is the MW of new 24-7 RA product that we must procure to close all hourly positions (in range [0, inf])
    ## 2. The hourly discharge Discharge[t] of our storage to close all hourly positions (in range [0, Storage_P])
    ## 3. The hourly charging Charge[t] of our storage to close all hourly positions (in range [0, Storage_P])
    ## 4. A binary variable State[t] that states whether the battery is charging or discharging in hour t, as it cannot be both
    ## There are 7 constraints:
    ## 1. For each hour t, NOP[t] - NewRA_P -  Discharge[t] + Charge[t] >= 0 (closed hourly SOD RA positions)
    ## 2. For all hours t, SUM(Discharge[t]) <= Storage_E (total storage discharge is <= total storage energy)
    ## 3. For all hours t, SUM(Discharge[t]) = Sum(Charge[t]) * 0.8 (total charge*RTE = total discharge)
    ## 4,5. For each hour t, charge[t] <= storage_cap and discharge[t] <= storage_cap
    ## 6,7. For each hour t, State[t] * Charge[t] = 0 and (1-State[t]) * Discharge[t] = 0 (ensures battery is either charging or discharging)

    # ra eligibility of generators
    mod.gen_is_ra_eligible = Param(mod.GENERATION_PROJECTS, within=Boolean)

    # hourly RA rquirements constraint
    mod.ra_sod_requirement  = Param(mod.PERIODS, mod.MONTHS, mod.HOURS, within = Reals)

    # cost of RA by technology type, WILL NEED UPDATING! for now only use single monthly cost for 24 hr gas RA
    mod.ra_sod_cost         = Param(mod.PERIODS, mod.MONTHS, within = NonNegativeReals)

    # resale value of RA by technology type, WILL NEED UPDATING! for now only use single monthly cost for 24 hr gas RA
    mod.ra_sod_resell_value = Param(mod.PERIODS, mod.MONTHS, within = NonNegativeReals, default = 0)

    # RA exceedance vals, WILL NEED UPDATING! for now just by tech type but should be by location too 
    mod.exceedance_vals     = Param(mod.PERIODS, mod.MONTHS, mod.HOURS, mod.GENERATION_TECHNOLOGIES_SOD, mod.REGIONS_SOD, within = Reals)

    # RA nqc
    mod.nqc_vals            = Param(mod.PERIODS, mod.MONTHS, mod.RESOURCE_ID_SOD, within = NonNegativeReals)

    # RA storage adders
    mod.storage_adders      = Param(mod.PERIODS, mod.MONTHS, within = NonNegativeReals, default = 0)

    # expression for RA qualifying capacity by generator, period, month, hour
    def CalculateHourlyQualifyingCapacities(m, g, p, mo, h):
        # g = generator, p = period, mo = month, tp = timepoint
        generator_qc = 0
        if m.gen_is_ra_eligible[g]:
            # if variable gen, pull month-hour nqc using capacity * exceedance value
            if m.gen_is_variable[g]:
                generator_qc = m.exceedance_vals[p, mo, h, m.gen_tech_sod[g], m.region_sod[g]] * m.GenCapacity[g, p]
            # if hydro, pull monthyl nqc from tab in model inputs 
            elif m.gen_tech_sod[g] == 'Small Hydro':
                generator_qc = m.nqc_vals[p, mo, m.resource_id_sod[g]] # need to add resource id
            # otherwise energy source should be geothermal or storage, and nqc is just the resource capacity for each hour
            else:
                generator_qc =  m.GenCapacity[g, p]
        return generator_qc
    
    mod.HourlyQC = Expression(
        mod.GENERATION_PROJECTS, mod.PERIODS, mod.MONTHS, mod.HOURS, rule = CalculateHourlyQualifyingCapacities
    )

    # expression for RA qualifying capacity by period, month, hour
    def CalculateEnergyNQC(m, p, mo, h):
        system_qc = 0
        for g in m.GENERATION_PROJECTS:
            # if storage project, don't add to system qc, will be added seperately to storage qc
            if m.gen_is_storage[g]:
                system_qc = system_qc
            # otherwise add hourly qc to system qc
            else:
                system_qc = system_qc + (
                    m.HourlyQC[g, p, mo, h] 
                )
        return system_qc 
    
    mod.EnergyNQC = Expression(
        mod.PERIODS, mod.MONTHS, mod.HOURS, rule = CalculateEnergyNQC
    )
    
    # expression for capacity of storage to be optimized for each period and month
    def CalculateStorageCapacity(m, p, mo):
        storage_cap = 0
        for g in m.GENERATION_PROJECTS:
            # if storage project, add to storage ap
            if m.gen_is_storage[g]:
                storage_cap = storage_cap + (
                    m.GenCapacity[g, p]
                )
            # otherwise pass
            else:
                storage_cap = storage_cap 
        # add storage cap from CAM and Contracts
        storage_cap = storage_cap + m.storage_adders[p, mo]
        return storage_cap 
    
    mod.StorageQC = Expression(
        mod.PERIODS, mod.MONTHS, rule = CalculateStorageCapacity
    )

    def CalculateStorageEnergy(m, p, mo):
        storage_energy = 0
        for g in m.GENERATION_PROJECTS:
            # if storage project, multiple capacity x storage_energy_to_cap ratio
            if m.gen_is_storage[g]:
                storage_energy = storage_energy + (
                    m.GenCapacity[g,p] * m.storage_energy_to_power_ratio[g]
                )
            # otherwise pass
            else:
                storage_energy = storage_energy
        # add storage energy from CAM and contracts
        storage_energy = storage_energy + m.storage_adders[p, mo] * 4
        return storage_energy
    
    mod.StorageEnergySOD = Expression(
        mod.PERIODS, mod.MONTHS, rule = CalculateStorageEnergy
    )


    # variable of storage discharge by period, month, hour
    mod.StorageDischarge    = Var(mod.PERIODS, mod.MONTHS, mod.HOURS, within = NonNegativeReals)

    # variable of storage charge by period, month, hour
    mod.StorageCharge       = Var(mod.PERIODS, mod.MONTHS, mod.HOURS, within = NonNegativeReals)

    # variable of storage state (binary, charging or discharging) by period, month, hour
    mod.StorageState        = Var(mod.PERIODS, mod.MONTHS, mod.HOURS, within = Binary)

    # variable of new 24-7 ra by period, month
    if mod.options.sell_excess_RA == "sell":
        mod.NewRA               = Var(mod.PERIODS, mod.MONTHS, within = Reals)
    else:
        mod.NewRA               = Var(mod.PERIODS, mod.MONTHS, within = NonNegativeReals)



    # expression for total storage dsicharge over all hours in a month
    mod.TotalMonthlyDischarge = Expression(
        mod.PERIODS, 
        mod.MONTHS,
        rule = lambda m, p, mo: sum(
            m.StorageDischarge[p, mo, h] for h in m.HOURS
        )
    )

    # expression for total storage dsicharge over all hours in a month
    mod.TotalMonthlyCharge = Expression(
        mod.PERIODS, 
        mod.MONTHS,
        rule = lambda m, p, mo: sum(
            m.StorageCharge[p, mo, h] for h in m.HOURS
        )
    )

    # CONSTRAINT: RA positions must be closed (period, month, hour)
    mod.RA_Position_Constraint = Constraint(
        mod.PERIODS,
        mod.MONTHS,
        mod.HOURS,
        rule = lambda m, p, mo, h: m.NewRA[p, mo] + m.StorageDischarge[p, mo, h] + m.EnergyNQC[p, mo, h] >= m.ra_sod_requirement[p, mo, h] + m.StorageCharge[p, mo, h]
    )

    # CONSTRAINT: total discharge <= storage_energy (period, month)
    mod.Storage_Energy_Constraint = Constraint(
        mod.PERIODS,
        mod.MONTHS,
        rule = lambda m, p, mo: m.TotalMonthlyDischarge[p, mo] <= m.StorageEnergySOD[p, mo] 
    )

    # CONSTRAINT: total discharge <= total charge * rte (period, month)
    mod.Storage_Charge_Discharge = Constraint(
        mod.PERIODS,
        mod.MONTHS,
        rule = lambda m, p, mo: m.TotalMonthlyDischarge[p, mo] == m.TotalMonthlyCharge[p, mo] * 0.8 # think about moving to model inputs... maybe in storage inputs tab
    )

    # CONSTRAINT: hourly discharge <= capacity (period, month, hour)
    mod.Hourly_Discharge_Constriant = Constraint(
        mod.PERIODS,
        mod.MONTHS,
        mod.HOURS,
        rule = lambda m, p, mo, h: m.StorageDischarge[p, mo, h] <= m.StorageQC[p, mo]
    )
    # CONSTRAINT: hourly charge <= capacity (period, month, hour)
    mod.Hourly_Charge_Constriant = Constraint(
        mod.PERIODS,
        mod.MONTHS,
        mod.HOURS,
        rule = lambda m, p, mo, h: m.StorageCharge[p, mo, h] <= m.StorageQC[p, mo]
    )

    # CONSTRAINT: storage state constraint #1, if state = 1 storage charge = 0 (period, month, hour)
    mod.Charging_State_Constraint = Constraint(
        mod.PERIODS,
        mod.MONTHS,
        mod.HOURS,
        rule = lambda m, p, mo, h: m.StorageState[p, mo, h] * m.StorageCharge[p, mo, h] == 0
    )

    # CONSTRAINT: storage state constraint #2, if state = 0 storage discharge = 0 (period, month, hour)
    mod.Discharging_State_Constraint = Constraint(
        mod.PERIODS,
        mod.MONTHS,
        mod.HOURS,
        rule = lambda m, p, mo, h: (1-m.StorageState[p, mo, h]) * m.StorageDischarge[p, mo, h] == 0
    )

    # expression for costs associated with new 24-hr RA by period, month
    mod.MonthlyRAOpenPositionCost = Expression(
        mod.PERIODS,
        mod.MONTHS,
        rule = lambda m, p, mo: m.NewRA[p, mo] * 1000 * m.ra_sod_cost[p, mo]
    )

    # expression for total RA costs by period
    mod.RAOpenPositionCost = Expression(
        mod.PERIODS,
        rule = lambda m, p: sum(
            m.MonthlyRAOpenPositionCost[p,mo] for mo in m.MONTHS
        )
    )

    # add RA cost to objective function
    mod.Cost_Components_Per_Period.append("RAOpenPositionCost")




def load_inputs(mod, match_data, inputs_dir):


    ## Info on projects in currnet porfolio
    match_data.load_aug(
        filename=os.path.join(inputs_dir, "generation_projects_info.csv"),
        auto_select=True,
        index=mod.GENERATION_PROJECTS,
        param=[mod.gen_is_ra_eligible],
    )

    ## SOD Obligations
    match_data.load_aug(
        filename = os.path.join(inputs_dir, "ra_sod_requirement.csv"),
        select = ("period", "month", "hour", "ra_requirement"),
        param = [mod.ra_sod_requirement]
    )

    ## SOD Exceedance values
    match_data.load_aug(
        filename=os.path.join(inputs_dir, "ra_exceedance_values.csv"),
        select=("period", "month", "hour", "gen_tech_sod", "region_sod", "exceedance_val"),
        param=[mod.exceedance_vals],
    )

    ## SOD NQCs
    match_data.load_aug(
        filename =os.path.join(inputs_dir, "ra_nqc_values.csv"),
        select=("period", "month",  "resource_id_sod", "nqc"),
        param=[mod.nqc_vals],
    )

    ## SOD RA costs
    match_data.load_aug(
        filename=os.path.join(inputs_dir, "ra_sod_costs.csv"),
        select=('period', 'month', 'ra_cost', 'ra_resell_value'),
        param=[mod.ra_sod_cost, mod.ra_sod_resell_value],
    )

    ## Additional Storage RA Capacity
    match_data.load_aug(
        filename = os.path.join(inputs_dir, "ra_sod_storage_adders.csv"),
        select = ('period', 'month', 'storage_ra_adders'),
        param =[mod.storage_adders]
    )



def post_solve(instance, outdir):
    
    ## Build output dataframe summarizing RA SOD results
    ra_dat = [
        {
            "Period": p,
            "RA_Requirement": "sod_RA",
            "Month": mo,
            "Hour": h,
            "RA_Requirement_Need_MW": value(instance.ra_sod_requirement[p, mo, h]),
            "Available_RA_Capacity_MW": value(instance.EnergyNQC[p, mo, h]),
            "RA_Position_MW": value(instance.EnergyNQC[p, mo, h] - instance.ra_sod_requirement[p, mo, h]),
            "Storage_Discharge_MW": value(instance.StorageDischarge[p, mo, h]),
            "Storage_Charge_MW": value(instance.StorageCharge[p, mo, h]),
            "New_24hr_RA_MW": value(instance.NewRA[p, mo]),
            "New_RA_Cost_$": value(instance.MonthlyRAOpenPositionCost[p, mo]),
        }
        for p in instance.PERIODS
        for mo in instance.MONTHS
        for h in instance.HOURS
        
    ]
    RA_df = pd.DataFrame(ra_dat)
    RA_df.set_index(["Hour", "Month", "Period"], inplace=True)

    RA_df.to_csv(os.path.join(outdir, "RA_summary.csv"))
