function NN = create_a_NN(algo)

    for i=1:size(algo.deep.sizes,2)-1
        NN.w{i} = randn(algo.deep.sizes(1,1+i), algo.deep.sizes(1,i));
        NN.b{i} = randn(algo.deep.sizes(1,1+i),1);
    end

    NN.sizes = algo.deep.sizes;

    NN.nbr_layers = size(NN.sizes,2);
    
    NN.dropout = algo.deep.dropout;

end
