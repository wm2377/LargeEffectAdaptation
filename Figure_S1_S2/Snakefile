import numpy as np
import os


N = 5000  # Population size
large_2NU_values = np.logspace(-3,3,7)[:-1]  # Large effect mutation rate values
sigma2_values = [4]
shift_values = [80]
individual_replicates = 100  # Number of replicates
all_allele_replicates = 100  # Number of replicates for all allele simulations

def get_replicate_expansion(wildcards):
    if wildcards.simulation_type == 'individual':
        replicates = individual_replicates
        return expand("/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/{{simulation_type}}_simulations/sigma2_{{sigma2}}/LargeN2U_{{large_2NU}}/shift_{{shift}}/processed_iteration_{n}_with_mutation_counts.pkl",
                    n=np.arange(0, replicates))
    else:
        replicates = all_allele_replicates
        return expand("/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/{{simulation_type}}_simulations/sigma2_{{sigma2}}/LargeN2U_{{large_2NU}}/shift_{{shift}}/processed_iteration_{n}.pkl",
                    n=np.arange(0, replicates))

# Define the final target rule, state which files you want to generate
rule all:
    input:
        expand("/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/all_alleles_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/better_skew_results.pkl",
                sigma2 = [4,40,80], shift=[50,80], large_2NU = [i for i in np.logspace(-3,3,13)[:-1]]),
        # expand("/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/{simulation_type}_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/all_processed_results_with_mutation_counts.pkl",
        #         sigma2 = [4,40,80], shift=[50,80], large_2NU = [i for i in np.logspace(-3,3,13)[:-1]], simulation_type = ['all_allele']),
        # expand("/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/{simulation_type}_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/all_processed_results.pkl",
        #         sigma2 = [4,40,80], shift=[50,80], large_2NU = , simulation_type = ['all_allele']),
        # expand("/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/{simulation_type}_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/all_done_with_all_mutation_counts.txt",
        #         sigma2 = [4], shift=[50,80], large_2NU = [i for i in np.logspace(-3,3,13)[:-1]], simulation_type = ['individual']),
        # # expand("/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/{simulation_type}_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/all_processed_results.pkl",
        # #        sigma2 = [4,40],shift=[80],large_2NU = [0.01,0.1,1,10], simulation_type = ['individual']),
        # expand("/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/{simulation_type}_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/all_processed_results_with_mutation_counts.pkl",
        #         sigma2 = [4], shift=[50,80], large_2NU = [i for i in np.logspace(-3,3,13)[:-2]], simulation_type = ['individual']),

# rule to run individual-based simulations
# burn in time and run time are in units of N generations
# sigma2 is in units of delta^2
# shift is in units of delta
rule run_individual_simulations:
    input:
        classes_file = "/insomnia001/home/wm2377/cdwm/large_effect_alleles/scripts/individual_classes.py"
    output:
        output_file = "/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/individual_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/all_done_with_all_mutation_counts.txt",
    resources:
        mem_mb_per_cpu = 500,
        time = 720,
        cpus_per_task=1,i8
    params:
        sigma2 = lambda wildcards: eval(wildcards.sigma2),
        large_2NU = lambda wildcards: eval(wildcards.large_2NU),
        N = N,
        shift = lambda wildcards: eval(wildcards.shift),
        burn_time = 10,
        run_time = 4,
        max_executor = 10,
        replicates = individual_replicates,
    script:
        "/insomnia001/home/wm2377/cdwm/large_effect_alleles/scripts/run_individual_simulations.py"

# rule to run all-allele simulations
rule run_all_allele_simulations:
    input:
        classes_file = "/insomnia001/home/wm2377/cdwm/large_effect_alleles/scripts/all_allele_classes.py"
    output:
        output_file = "/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/all_allele_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/all_done.txt",
    resources:
        mem_mb_per_cpu = 500,
        time = 720,
        cpus_per_task = 10,
        ntasks=1,
    params:
        sigma2 = lambda wildcards: eval(wildcards.sigma2),
        large_2NU = lambda wildcards: eval(wildcards.large_2NU),
        N = N,
        shift = lambda wildcards: eval(wildcards.shift),
        burn_time = 10,
        run_time = 4,
        max_executor = 10,
        replicates = all_allele_replicates,
    script:
        "/insomnia001/home/wm2377/cdwm/large_effect_alleles/scripts/run_all_allele_simulations.py"
        
# rule to process results from individual-based or all-allele simulations
rule process_simulation_iteration_results:
    input:
        all_done = "/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/{simulation_type}_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/all_done.txt",
        simulation_results = "/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/{simulation_type}_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/iteration_{n}.pkl",
        population = "/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/{simulation_type}_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/iteration_{n}_population.pkl",
    output:
        processed_results = "/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/{simulation_type}_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/processed_iteration_{n}.pkl",
    params:
        simulation_type = lambda wildcards: 'individual' if 'individual' in wildcards.simulation_type else 'all_allele',
    resources:
        mem_mb_per_cpu = 500,
        time = 60,
        cpus_per_task=1,
        ntasks=1,
    script:
        "/insomnia001/home/wm2377/cdwm/large_effect_alleles/scripts/process_simulation_iteration_results.py"
        
# rule to combine processed results from all replicates into a single file
rule process_all_individual_simulation_results:
    input:
        simulation_results = lambda wildcards: get_replicate_expansion(wildcards)
    output:
        processed_results = "/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/{simulation_type}_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/all_processed_results_with_mutation_counts.pkl",
    params:
        simulation_type = lambda wildcards: 'individual' if 'individual' in wildcards.simulation_type else 'all_allele',
    resources:
        mem_mb_per_cpu = 5000,
        time = 120,
        cpus_per_task=1,
        ntasks=1,
    script:
        "/insomnia001/home/wm2377/cdwm/large_effect_alleles/scripts/process_results.py"

# I used this to calculate the skew statistics for all alleles, because it was easier to do it in a separate script and something was not working with the above processing rules
rule process_skew_all_alleles:
    input:
        simulation_results = '/insomnia001/depts/pas_lab/users/wm2377/large_effect_alleles/scripts/calculate_skew_all_alleles.py',
    output:
        processed_results = "/insomnia001/home/wm2377/cdwm/large_effect_alleles/data/all_alleles_simulations/sigma2_{sigma2}/LargeN2U_{large_2NU}/shift_{shift}/better_skew_results.pkl",
    params:
        sigma2 = lambda wildcards: eval(wildcards.sigma2),
        large_2NU = lambda wildcards: eval(wildcards.large_2NU),
        shift = lambda wildcards: eval(wildcards.shift),
    resources:
        mem_mb_per_cpu = 512,
        time = 720,
        cpus_per_task=1,
        ntasks=1,
    script:
        "/insomnia001/depts/pas_lab/users/wm2377/large_effect_alleles/scripts/calculate_skew_all_alleles.py"
