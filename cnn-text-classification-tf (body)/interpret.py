import numpy as np

def interpret_many(x_raw, relu, pool, best_trigrams = {}):
	#print(pool.shape, "is pool shape ")
	pool = pool.squeeze() #should be len(x_raw) x num_filters (128)
	relu = relu.squeeze()
	text_lists = []
	for text in x_raw:
		text_list = text.split()
		text_lists.append(text_list)
	#print(pool.shape, "is pool shape ")
	#print(relu.shape, "is relu shape")
	#print(np.array(x_raw).shape, "is xraw shape")
	for i in range(pool.shape[0]):
		best_trigrams = interpret(text_lists[i],relu[i], pool[i], best_trigrams)
	#print(best_trigrams)
	return best_trigrams


def find_relu_index(relu, pooled_val, i):
	#each thing in relu should be 128 length
	#index represents the index (out of max seq len) that resulted in the pooled val
	for ind, arr in enumerate(relu):
		if arr[i]==pooled_val:
			return ind
	return None



def interpret(word_list, relu, pool, best_trigrams={}):
	#print(relu.shape, "is relu in interpret")
	#print(relu.squeeze().shape, "is relu squeeze")
	relu = relu.squeeze()
	for i, pooled_val in enumerate(pool):
		relu_index = find_relu_index(relu, pooled_val, i)
		trigram = word_list[relu_index:relu_index+3]
		#print("trigram: ",trigram)
		#print("i: ", i)
		if i in best_trigrams:
			best_trigrams[i]+=[(pooled_val,[trigram])]
		else:
			best_trigrams[i]=[(pooled_val,[trigram])]
	return best_trigrams

def get_best_n_for_each_neuron(best_trigrams,n):
	best_n_trigrams=best_trigrams.copy()
	for neuron in best_trigrams.keys():
		best_n_trigrams[neuron]=sorted(best_trigrams[neuron])[:10]
	return best_n_trigrams
