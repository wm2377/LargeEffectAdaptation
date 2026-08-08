import pickle
from collections import defaultdict

if snakemake.params.simulation_type == 'individual':
    import individual_classes as classes
else:
    import all_allele_classes as classes
    
# Process the results from a single iteration of the simulation
# For each time point, calculate the mean, variance, and skew of the genetic variance contributed
# by alleles in different effect size bins
def process_data_for_given_time_and_iteration(mutations_dict,N=5000,bin_limits=[100,20000]):
    results = {bins:{'mean':0,
                     'variance':0,
                     'skew':0} for bins in bin_limits}
    if snakemake.params.simulation_type == 'individual':
        for mut,count in mutations_dict.items():
            a2 = mut.a2
            x = count/(2*N)
            for bin_value in bin_limits:
                if a2 < bin_value:
                    results[bin_value]['mean'] += mut.a*2*x
                    results[bin_value]['variance'] += 2*mut.a2*x*(1-x)
                    results[bin_value]['skew'] += mut.a**3*(1-x)*(1-2*x)*x
                    break
    else:
        for a,x in mutations_dict:
            a2 = a**2
            a3 = a**3
            for bin_value in bin_limits:
                if a2 < bin_value:
                    results[bin_value]['mean'] += a*2*x
                    results[bin_value]['variance'] += 2*a2*x*(1-x)
                    results[bin_value]['skew'] += a3*(1-x)*(1-2*x)*x
                    break
    return results
    
def main():
    
    filepath = snakemake.input.simulation_results
    results = {}
    with open(filepath,'rb') as infile:
        while True:
            try:
                data = pickle.load(infile)
                for key in data:
                    
                    if snakemake.params.simulation_type == 'individual':
                        results[key] = process_data_for_given_time_and_iteration(data[key][1])
                    else:
                        results[key] = process_data_for_given_time_and_iteration(data[key]['mutations'])
            except EOFError:
                break
            
    with open(snakemake.input.population, 'rb') as popfile:
        population = pickle.load(popfile)
    fixations = population[0].fixations
    fixations = [m.a for m in fixations]
        
    output_filepath = snakemake.output.processed_results
    with open(output_filepath,'wb') as outfile:
        pickle.dump((results,fixations),outfile)

if __name__ == "__main__":
    main()