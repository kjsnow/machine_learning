import config
import helper_functions as hf

def main():
	# Read hyperparameters from config file and create Hyperparameters object
	params = hf.Hyperparameters(experiment=config.experiment,
								query=config.query,
								col_list=config.col_list, 
								num_nodes=config.num_nodes, 
								num_epochs=config.num_epochs, 
								learning_rate=config.learning_rate,
								last_day_testing=config.last_day_testing,
								storage_path=config.storage_path,
								to_query=config.to_query,
								save_results=config.save_results,
								raw_file_path=config.raw_file_path)
	
	print('Params created')

	# Run query or load file from folder
	sample = hf.fetch_data(params=params, credentials=config.creds)

	print('Fetch complete, setting up inputs')

	# Split training and testing
	# if last_day_testing == True, last day of churn will be testing set, otherwise shuffle
	inputs = hf.shuffle_split(sample, last_day_testing=params.last_day_testing)

	print('Inputs ready, running model')

	hf.ann(params=params, inputs=inputs)

if __name__ == '__main__':
    main()
