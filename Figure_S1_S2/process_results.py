import pickle
from collections import defaultdict
if snakemake.params.simulation_type == 'individual':
    import individual_classes as classes
else:
    import all_allele_classes as classes
    
# aggregate results from multiple simulation iterations
def process_results(iteration_results_dict):
    processed_results = {}
    for iteration in iteration_results_dict:
        for key in iteration_results_dict[iteration]:
            if key not in processed_results:
                processed_results[key] = {}
            for bin_value in iteration_results_dict[iteration][key]:
                if bin_value not in processed_results[key]:
                    processed_results[key][bin_value] = {'mean': {'sum': 0, 'sum_sq': 0, 'count': 0},
                                                         'variance': {'sum': 0, 'sum_sq': 0, 'count': 0},
                                                         'skew': {'sum': 0, 'sum_sq': 0, 'count': 0}}
                processed_results[key][bin_value]['mean']['sum'] += iteration_results_dict[iteration][key][bin_value]['mean']
                processed_results[key][bin_value]['mean']['sum_sq'] += iteration_results_dict[iteration][key][bin_value]['mean'] ** 2
                processed_results[key][bin_value]['mean']['count'] += 1
                processed_results[key][bin_value]['variance']['sum'] += iteration_results_dict[iteration][key][bin_value]['variance']
                processed_results[key][bin_value]['variance']['sum_sq'] += iteration_results_dict[iteration][key][bin_value]['variance'] ** 2
                processed_results[key][bin_value]['variance']['count'] += 1
                processed_results[key][bin_value]['skew']['sum'] += iteration_results_dict[iteration][key][bin_value]['skew']
                processed_results[key][bin_value]['skew']['sum_sq'] += iteration_results_dict[iteration][key][bin_value]['skew'] ** 2
                processed_results[key][bin_value]['skew']['count'] += 1
    return processed_results

def main():
    
    filepaths = snakemake.input.simulation_results
    iteration_results = {}
    iteration_fixations = {}
    for filepath in filepaths:
        with open(filepath,'rb') as infile:
            temp_results,fixation   = pickle.load(infile)
            iteration_results[filepath] = temp_results
            iteration_fixations[filepath] = fixation
        
    processed_results = process_results(iteration_results)
    output_filepath = snakemake.output.processed_results
    with open(output_filepath,'wb') as outfile:
        pickle.dump(processed_results,outfile)
        pickle.dump(iteration_fixations,outfile)
        
if __name__ == "__main__":
    main()