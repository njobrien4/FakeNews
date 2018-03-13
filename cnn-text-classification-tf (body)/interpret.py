def interpret(word_list, relu, pool):
	for i, pooled_val in enumerate(pool):
		relu_index = find_relu_index(relu, pooled_val, i)
		trigram = word_list[relu_index:relu_index+3]
		print("trigram: ",trigram)
		print("i: ", i)
		