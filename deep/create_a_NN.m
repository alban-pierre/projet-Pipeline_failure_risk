function NN = create_a_NN(sizes)

	for i=1:size(sizes,2)-1
		NN.w{i} = randn(sizes(1,1+i), sizes(1,i));
		NN.b{i} = randn(sizes(1,1+i),1);
	end

	NN.sizes = sizes;

	NN.nbr_layers = size(NN.sizes,2);
	
	NN.dropout = 0.7;

end
