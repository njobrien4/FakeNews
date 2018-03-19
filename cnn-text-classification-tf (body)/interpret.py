import numpy as np
#import matplotlib.pyplot as plt
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

def make_weight_histogram(weights):
	t1 = np.arange(0.0, 5.0, 0.1)
	t2 = np.arange(0.0, 5.0, 0.02)

	plt.figure(1)
	plt.subplot(211)
	plt.title("Weights for Fake News Indicator")
	plt.hist(weights[:,0], bins = 20, range = [-0.3,0.3])

	plt.subplot(212)
	plt.title("Weights for Real News Indicator")
	plt.hist(weights[:,1], bins=20, range = [-0.3,0.3])
	plt.show()

	# from data helpers: 
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]

def get_n_best_neurons(weights, n,abs_value = False):
	print(weights, weights.shape)
	arr_0 = weights[:,0]
	list_0=arr_0.argsort()[-n:][::-1]
	list_0_neg = arr_0.argsort()[:n]
	arr_1 = weights[:,1]
	list_1=arr_1.argsort()[-n:][::-1]
	list_1_neg = arr_0.argsort()[:n]
	#return weights for fake news, weights for real news
	return list_0, list_1, list_0_neg, list_1_neg

