function NN = train_a_NN(NN, datax, datay, learn_rate)

    sigmoid = @(x) (1./(1+exp(-x)));

    %dsigmoid = @(x) (sigmoid(x).*(1-sigmoid(x)));

    for j=1:size(datax,1)
        NN.a{1} = datax(j,:)';
    
        for i=1:NN.nbr_layers-1
            NN.z{i} = NN.w{i}*NN.a{i} + NN.b{i};
            NN.a{i+1} = sigmoid(NN.z{i});
        end
        
        NN = update_a_NN(NN, datay(j,:), learn_rate);
    end
    
end
