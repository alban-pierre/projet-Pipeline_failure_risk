function test_a_NN(NN, algo, trainx, trainy, printall)

    % Function that says if the neural network learns correctly, based on a learning example by example (no batchs)

    % Dimensions :
    % N  : Number of training examples
    % Dx : Dimension of examples
    % Dy : Dimension of prediction

    % Input :
    % NN       : (structure) : The neural network, containing coefficients, some parameters, etc
    % algo     : (structure) : All parameters that defines an algorithm
    % trainx   : (N*Dx)      : Training set
    % trainy   : (N*Dy)      : Training output
    % printall : (1*1)       : Set it to one to print all neural network parameters derivative dw and db
    
    if (nargin < 5)
        printall = 0;
    end

    ma = 1;
    mi = 1;
    epss = 0.000001;
    
    for itr = 1:size(trainx,1)
        clear w;
        NN = feed_forward_one(NN, trainx(itr,:));
        for l=1:size(NN.w,2)
            for i=1:size(NN.w{l},1)
                for j=1:size(NN.w{l},2)
                    Nn = NN;
                    Nn.w{l}(i,j) += epss;
                    Nn = feed_forward_one(Nn, trainx(itr,:));
                    if (algo.deep.costfunction == 1)
                        w{l}(i,j) = ((Nn.a{Nn.nbr_layers}' - trainy(itr,:)).^2 - (NN.a{NN.nbr_layers}' - trainy(itr,:)).^2)*[0.6;0.4];
                    else
                        w{l}(i,j) = (trainy(itr,:).*(log(Nn.a{Nn.nbr_layers})') + (1-trainy(itr,:)).*(log(1-Nn.a{Nn.nbr_layers})'))*[1;1] - (trainy(itr,:).*(log(NN.a{NN.nbr_layers})') + (1-trainy(itr,:)).*(log(1-NN.a{NN.nbr_layers})'))*[1;1];
                    end
                end
            end
        end
        clear b;
        for l=1:size(NN.b,2)
            for i=1:size(NN.b{l},1)
                Nn = NN;
                Nn.b{l}(i,1) += epss;
                Nn = feed_forward_one(Nn, trainx(itr,:));
                if (algo.deep.costfunction == 1)
                    b{l}(i,1) = ((Nn.a{Nn.nbr_layers}' - trainy(itr,:)).^2 - (NN.a{NN.nbr_layers}' - trainy(itr,:)).^2)*[0.6;0.4];
                else
                    b{l}(i,1) = (trainy(itr,:).*(log(Nn.a{Nn.nbr_layers})') + (1-trainy(itr,:)).*(log(1-Nn.a{Nn.nbr_layers})'))*[1;1] - (trainy(itr,:).*(log(NN.a{NN.nbr_layers})') + (1-trainy(itr,:)).*(log(1-NN.a{NN.nbr_layers})'))*[1;1];
                end
            end
        end
        algo.deep.uniformbatch = 1;
        algo.deep.batchsize = 1;
        algo.deep.dropout = 0;
        algo.deep.regularization = 0;
		NNN = NN;
        NN = train_a_NN(NN, algo, trainx(itr,:), trainy(itr,:), 1);
        for l=1:size(NN.w,2)
            a = NN.w{l} - NNN.w{l};
            if (printall)
                a./w{l}*epss
				mean(mean(abs(a - w{l}/epss)))
            end
            ma = max(max(max(a./w{l}*epss)), ma);
            mi = min(min(min(a./w{l}*epss)), mi);
        end
        for l=1:size(NN.b,2)
            a = NN.b{l} - NNN.b{l};
            if (printall)
                a./b{l}*epss
				mean(mean(abs(a - b{l}/epss)))
            end
            ma = max(max(max(a./b{l}*epss)), ma);
            mi = min(min(min(a./b{l}*epss)), mi);
        end
        
        NN = feed_forward_one(NN, trainx(itr,:));
        NNN = feed_forward_one(NNN, trainx(itr,:));
        if (algo.deep.costfunction == 1)
            if (((NNN.a{Nn.nbr_layers}' - trainy(itr,:)).^2 - (NN.a{NN.nbr_layers}' - trainy(itr,:)).^2)*[0.6;0.4] < 0)
                fprintf(2, 'Good direction\n');
            else
                fprintf(2, 'Bad direction\n');
            end
        else
            if ((trainy(itr,:).*(log(NNN.a{NNN.nbr_layers})') + (1-trainy(itr,:)).*(log(1-NNN.a{NNN.nbr_layers})'))*[1;1] - (trainy(itr,:).*(log(NN.a{NN.nbr_layers})') + (1-trainy(itr,:)).*(log(1-NN.a{NN.nbr_layers})'))*[1;1] < 0)
                fprintf(2, 'Good direction\n');
            else
                fprintf(2, 'Bad direction\n');
            end
        end
    end
    fprintf(2, 'All errors are between %f and %f\n', mi-1, ma-1);

end
