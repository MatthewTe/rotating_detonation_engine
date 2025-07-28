from chemicals.combustion import combustion_stoichiometry
from chempy import balance_stoichiometry, Equilibrium
from chempy.chemistry import Reaction, mass_fractions
import pprint
from sympy import symbols, solve, Eq

import sympy
from loguru import logger

def calculate_combustion_chemical_properties():

    # CH4 + 2O2 -> CO2 + 2H2O
    reac, prod =  balance_stoichiometry({"O2", "CH4"}, {"CO2", "H2O"}, underdetermined=None)

    # If we assume equivalence ratio is oxygen limited:
    equivalence_ratio = 1.65

    oxidizer_ratio = reac["O2"] / equivalence_ratio
    fuel_ratio = reac['CH4']

    # Solve the chemical equation by balancing the coeffs as a system of linear equations for the atomic balances:
    # (1)CH4 + (1.212)O2 -> (a)CO2 + (b)H2O + (c)C: 
    a, b, c = symbols("a b c")
    carbon_eq = Eq(1, a + c)
    oxygen_eq = Eq(oxidizer_ratio*2, 2*a + b)
    hydrogen_eq = Eq(4, b)
    
    solution = solve([carbon_eq, oxygen_eq, hydrogen_eq], [a, b, c], dict=True)
    
    # mass fractions of each reactant Yi:
    Yi = mass_fractions({"CH4": 1, "O2": oxidizer_ratio})
    #print(Yi)

    # specific gas constants (mass weighed averages):
    # CH4
    R_methane =  518.28
    R_oxygen =  259.84
    R_mix = (R_methane * Yi['CH4']) + (R_oxygen * Yi['O2'])
    #print(R_mix)

    # Ratio of specific heats based on mass fraction:
    λ_methane = 1.32
    λ_oxygen = 1.40
    λ_mix = (λ_methane * Yi["CH4"]) + (λ_oxygen * Yi['O2'])
    #print(λ_mix)

    return {"Yi": Yi, "R_mix": R_mix, "specific_heats": λ_mix} 

def solve_mass_flow_rate_equation(**kwargs):

    # Declare symbols:
    m_dot, A_star_mm, P0_kPa, ymix, Rs_mix_J_Kg_K, T0_Kelvin = symbols("m A* P0 γ Rs_mix T0")
    mass_flow_expression = sympy.Eq(
        m_dot, 
        A_star_mm * P0_kPa * sympy.sqrt(
            (   
                (  ymix / (Rs_mix_J_Kg_K * T0_Kelvin) ) * 
                ( (ymix + 1) / 2 ) ** 
                ( (ymix + 1) / (1 - ymix))
            ) 
        )
    )

    logger.info(f"Mass flow rate equation through a chocked orifice:")
    sympy.pprint(mass_flow_expression)

    for k, v in kwargs.items():
        match k:
            case "m_dot":
                mass_flow_expression = mass_flow_expression.subs(m_dot, v)
            case "A_star":
                mass_flow_expression = mass_flow_expression.subs(A_star_mm, v)
            case "P0":
                mass_flow_expression = mass_flow_expression.subs(P0_kPa, v)
            case "ymix":
                mass_flow_expression = mass_flow_expression.subs(ymix, v)
            case "Rs_mix":
                mass_flow_expression = mass_flow_expression.subs(Rs_mix_J_Kg_K, v)
            case "T0":
               mass_flow_expression = mass_flow_expression.subs(T0_Kelvin, v)

    # mass_flow_expression.subs()
    result_set = sympy.solveset(mass_flow_expression, symbol=A_star_mm)
    return result_set

def solve_mach_number_of_injected_fuel(**kwargs):

    A, A_star, ymix, Mach_inj = symbols("A A* γ Minj")
    root_finding_mach_eq = Eq(
        A/A_star,
        (2/ (ymix + 1)) ** ((ymix + 1)/(ymix - 1)) * 
        (1/Mach_inj) * sympy.sqrt((
            1 + ( ((ymix -1)/2) ) * Mach_inj**2) ** ((ymix + 1)/ (ymix - 1)))
    )
    
    logger.info(f"Root finding the Mach number via the compressible area ratio:")
    sympy.pprint(root_finding_mach_eq)

    for k, v in kwargs.items():

        match k:
            case "A":
                root_finding_mach_eq = root_finding_mach_eq.subs(A, v)
            case "A_star":
                root_finding_mach_eq = root_finding_mach_eq.subs(A_star, v)
            case "ymix":
                root_finding_mach_eq = root_finding_mach_eq.subs(ymix, v)
            case "Mach_inj":
                root_finding_mach_eq = root_finding_mach_eq.subs(Mach_inj, v)

    assert root_finding_mach_eq.free_symbols == {Mach_inj}, f"Unresolved symbols: {root_finding_mach_eq.free_symbols}"

    subsonic_solution = sympy.nsolve(root_finding_mach_eq, Mach_inj, 0.01)
    supersonic_solution = sympy.nsolve(root_finding_mach_eq, Mach_inj, 7.5)

    return {"subsonic": subsonic_solution, "supersonic": supersonic_solution}

def solve_injector_static_pressure(**kwargs):

    P_inj, P0, ymix, M = symbols("Pinj P0 γ M")
    isentropic_p_ratio_eq = Eq(
        P_inj/P0,
        (1 + ((ymix -1)/ 2) * M**2) ** (ymix / (1 - ymix))
    )

    logger.info(f"Calculating static pressure of injected fuel via the isentropic pressure ratio equation:")
    sympy.pprint(isentropic_p_ratio_eq)

    for k, v in kwargs.items():

        match k:
            case "P_inj":
                isentropic_p_ratio_eq = isentropic_p_ratio_eq.subs(P_inj, v)
            case "P0":
                isentropic_p_ratio_eq = isentropic_p_ratio_eq.subs(P0, v)
            case "ymix":
                isentropic_p_ratio_eq = isentropic_p_ratio_eq.subs(ymix, v)
            case "M":
                isentropic_p_ratio_eq = isentropic_p_ratio_eq.subs(M, v)

    solution = sympy.solveset(isentropic_p_ratio_eq, symbol=P_inj)
    return solution

def solve_pressure_ratio_equation(**kwargs):

    P_cc, P_inj, ymix, M = symbols("Pcc P_inj γ M")
    pressure_ratio_eq = Eq(
        P_cc/P_inj,
        1 + ( (2 * ymix) / (ymix + 1) ) * (M**2 - 1)
    )

    logger.info(f"Calculating annulus pressure using pressure ratio equation:")
    sympy.pprint(pressure_ratio_eq)

    for k, v in kwargs.items():

        match k:
            case "P_cc":
                pressure_ratio_eq = pressure_ratio_eq.subs(P_cc, v)
            case "P_inj":
                pressure_ratio_eq = pressure_ratio_eq.subs(P_inj, v)
            case "ymix":
                pressure_ratio_eq = pressure_ratio_eq.subs(ymix, v)
            case "M":
                pressure_ratio_eq = pressure_ratio_eq.subs(M, v)

    solution = sympy.solveset(pressure_ratio_eq, symbol=P_cc)
    return solution 

if __name__ == "__main__":

    sympy.init_printing()

    combustion_properties = calculate_combustion_chemical_properties()
    SPECIFIC_GAS_CONSTANT_MIXTURE = combustion_properties['R_mix']
    RATIO_SPECIFIC_HEATS_MIXTURE = combustion_properties['specific_heats']
    

    ENGINE_OUTER_DIAMETER_MM = 150
    STAGNATION_TEMPERATURE_K = 293
    ANNULUS_THICKNESS_MM = 5
    TOTAL_PRESSURE_KPA = 700
    MASS_FLOW_RATE_KG_S = 0.210
    
    RDE_ANNULUS_AREA_REQUIRED_TO_CHOKE_FLOW = solve_mass_flow_rate_equation(
        m_dot=MASS_FLOW_RATE_KG_S,
        P0=TOTAL_PRESSURE_KPA,
        ymix=RATIO_SPECIFIC_HEATS_MIXTURE,
        Rs_mix=SPECIFIC_GAS_CONSTANT_MIXTURE,
        T0=STAGNATION_TEMPERATURE_K
    )

    INJECTED_FUEL_MACH_NUMBER = solve_mach_number_of_injected_fuel(
        A=ENGINE_OUTER_DIAMETER_MM,
        A_star=next(iter(RDE_ANNULUS_AREA_REQUIRED_TO_CHOKE_FLOW)),
        ymix=RATIO_SPECIFIC_HEATS_MIXTURE
    )

    INJECTED_FUEL_STATIC_PRESSURE = solve_injector_static_pressure(
        P0=TOTAL_PRESSURE_KPA,
        ymix=RATIO_SPECIFIC_HEATS_MIXTURE,
        M=INJECTED_FUEL_MACH_NUMBER['supersonic']
    )

    ANNULUS_POST_SHOCK_PRESSURE = solve_pressure_ratio_equation(
        P_inj=next(iter(INJECTED_FUEL_STATIC_PRESSURE)),
        ymix=RATIO_SPECIFIC_HEATS_MIXTURE,
        M=INJECTED_FUEL_MACH_NUMBER['supersonic']
    )

    print(ANNULUS_POST_SHOCK_PRESSURE)