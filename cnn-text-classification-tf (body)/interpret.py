import numpy as np
#import matplotlib.pyplot as plt
def interpret_many(x_raw, relu, pool, all_wi_ai, best_trigrams = {}, n=5):
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
	top_n_neurons = []
	for i in range(pool.shape[0]):
		best_trigrams = interpret(text_lists[i],relu[i], pool[i], best_trigrams)
		weights = all_wi_ai[i].T #2 x 128 --> 128 x 2
		top_n_neurons.append([get_n_best_neurons(weights,5)])
	#print(best_trigrams)
	return best_trigrams, top_n_neurons


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
		best_n_trigrams[neuron]=sorted(best_trigrams[neuron])[:n]
	return best_n_trigrams

def make_weight_histogram(weights):
	plt.figure(1)
	plt.subplot(211)
	plt.title("Weights for Fake News Indicator")
	plt.hist(weights[:,0], bins = 20, range = [-0.3,0.3])

	plt.subplot(212)
	plt.title("Weights for Real News Indicator")
	plt.hist(weights[:,1], bins=20, range = [-0.3,0.3])
	plt.show()

def get_most_relevant_neurons(all_wi_ai=None, ind = None, abs = False):
	pass
	if ind is None:
		fake_news = np.mean(all_wi_ai[:,0,:], axis = 0)
		real_news = np.mean(all_wi_ai[:,1,:], axis = 1)


def make_wi_ai_histogram(all_wi_ai, ind = None):
	if ind is None:
		#plot the average
		fake_news = np.mean(all_wi_ai[:,0,:], axis = 0)
		real_news = np.mean(all_wi_ai[:,1,:], axis = 1)
	else:
		wi_ai = all_wi_ai[ind]
		#plot x_raw[ind] weights*activation for fake news indicator
		fake_news=wi_ai[0]
		#plot x_raw[ind] weights*activation for real news indicator
		real_news=wi_ai[1]
	plt.figure(1)
	pt.subplot(211)
	plt.title("W_i * a_i for Fake News Indicator")
	plt.hist(fake_news)

	plt.subplot(212)
	plt.title("W_i * a_i for Real News Indicator")
	plt.hist(real_news)
	plt.show()



	# from data helpers: 
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]

def get_n_best_neurons(weights, n,abs_value = False):
	#print(weights, weights.shape) #128 x 2
	arr_0 = weights[:,0]
	list_0=arr_0.argsort()[-n:][::-1]
	list_0_neg = arr_0.argsort()[:n]
	arr_1 = weights[:,1]
	list_1=arr_1.argsort()[-n:][::-1]
	list_1_neg = arr_0.argsort()[:n]
	#return weights for fake news, weights for real news
	return list_0, list_1, list_0_neg, list_1_neg

def get_info(ind, all_wi_ai, all_top_neurons):
	cur_dir = "log/"
	import pickle
	with open(cur_dir+all_top_neurons, 'rb') as f:
		all_top_neurons = pickle.load(f) #all top neurons has most relevant neurons for each x_raw
	print(all_top_neurons[0], "is all top neurons[0]")
	print(len(all_top_neurons), "is len all top neuruons")
	print(all_top_neurons[ind], "is all top neurons[ind]")
	make_wi_ai_histogram(cur_dir+all_wi_ai, ind)


