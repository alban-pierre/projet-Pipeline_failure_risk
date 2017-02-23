function NN = update_a_NN(NN, algo, datax, datay, learn_rate)

    % Function that updates parameters of the neural network, based on a batch of examples

    % Dimensions :
    % N  : Number of examples in the batch
    % Dx : Dimension of examples
    % Dy : Dimension of prediction

    % Input :
    % NN         : (structure) : The neural network, containing coefficients, some parameters, etc
    % algo       : (structure) : All parameters that defines an algorithm
    % trainx     : (N*Dx)      : Training set
    % trainy     : (N*Dy)      : Training output
    % learn_rate : (1*1)       : The learning rate of our neural network

    % Output :
    % NN : (structure) : The neural network, containing coefficients, some parameters, etc

    sigmoid = @(x) (1./(1+exp(-x)));

    for i=NN.nbr_layers-1:-1:1
        dCdb{i} = zeros(size(NN.b{i}));
        dCdw{i} = zeros(size(NN.w{i}));
    end
    
    for j=1:size(datax,1)
    
        NN = feed_forward_one(NN, datax(j,:));
        
        if (algo.deep.costfunction == 1)
            % In the case of a square cost function 
            dCda = 2*(NN.a{NN.nbr_layers}' - datay(j,:)).*[0.6,0.4];
            for i=NN.nbr_layers-1:-1:1
                dadb{i} = NN.a{i+1}.*(1-NN.a{i+1});
                dada{i} = NN.w{i}.*dadb{i};
            end            
            % Computes derivatives of C
            for i=NN.nbr_layers-1:-1:1
                assert(size(dCda, 2) == size(dadb{i},1));
                dCdb{i} = dCdb{i} + dCda'.*dadb{i};
                dCdw{i} = dCdw{i} + (dCda'.*dadb{i})*(NN.a{i}');
                if (i>1)
                    dCda = dCda*dada{i};
                end
            end
        else
            % In the case of a cross-entropy cost function
            for i=NN.nbr_layers-2:-1:1
                dadb{i} = NN.a{i+1}.*(1-NN.a{i+1});
                dada{i} = NN.w{i}.*dadb{i};
            end
            d = datay(j,:)' - NN.a{NN.nbr_layers};
            dCdb{NN.nbr_layers-1} = dCdb{NN.nbr_layers-1} + d;
            dCdw{NN.nbr_layers-1} = dCdw{NN.nbr_layers-1} + d*(NN.a{NN.nbr_layers-1}');
            dCda = d'*NN.w{NN.nbr_layers-1};
            % Computes derivatives of C
            for i=NN.nbr_layers-2:-1:1
                assert(size(dCda, 2) == size(dadb{i},1));
                dCdb{i} = dCdb{i} + dCda'.*dadb{i};
                dCdw{i} = dCdw{i} + (dCda'.*dadb{i})*(NN.a{i}');
                if (i>1)
                    dCda = dCda*dada{i};
                end
            end
        end
        
    end

    % Divide the gradient by the batch size
    for i=NN.nbr_layers-1:-1:1
        dCdb{i} = dCdb{i}./size(datax,1);
        dCdw{i} = dCdw{i}./size(datax,1);
    end
    
    % Update parameters
    for i=NN.nbr_layers-1:-1:1
        NN.b{i} = NN.b{i} + learn_rate*dCdb{i};
        NN.w{i} = NN.w{i} + learn_rate*dCdw{i};
    end
    
end
