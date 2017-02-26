function repr = neural_network_representation(deep, trainx, trainy)

    % Prediction based on neural network, trained from scratch

    % Dimensions :
    % N : Number of training examples
    % Dx : Dimension of examples
    % Dy : Dimension of prediction
    % D  : Dimension of the representation, equal to the size of the before last layer
    
    % Input :
    % deep   : (structure) : Parameters of the neural network
    % trainx : (N*Dx)      : Training set
    % trainy : (N*Dy)      : Training output
    
    % Output :
    % repr : (N*D) : The neural network representation wanted
    
    assert(nargin >= 3, 'Error : neural_network_reprentation requires at least 3 arguments : NN parameters, trainx, trainy');
    assert(size(trainx,1) == size(trainy,1), 'trainx and trainy must contain the same number of examples');
    
    [N, Dx] = size(trainx);
    Dy = size(trainy, 2);
    
    algoo.deep = deep;

    NN = create_a_NN(algoo);

    if (deep.show_epoch_err ~= 0)
        err_tr = zeros(deep.epoch,Dy);
        for kk = 1:deep.epoch
            r = randperm(N);
            NN = train_a_NN(NN, algoo, trainx(r,:), trainy(r,:), deep.learn_rate(kk));
            [NN, a] = feed_forward_several(NN, trainx);
            prediction_train = a{end}';
            [err_tr(kk,:), ~] = auc_error(prediction_train, trainy);
            fprintf(2,'Ep%d ', kk);
        end
        fprintf(2,'\n');
        
        figure(deep.show_epoch_err); hold on;
        plot(1:size(err_tr,1), err_tr(:,1)', 'k-');
        plot(1:size(err_tr,1), err_tr(:,2)', 'r-');
        legend('train 2014', 'train 2015', 'location', 'southeast');
    else
        for kk = 1:deep.epoch
            r = randperm(N);
            NN = train_a_NN(NN, algoo, trainx(r,:), trainy(r,:), deep.learn_rate(kk));
            fprintf(2,'Ep%d ', kk);
        end
        fprintf(2,'\n');
    
        [NN, a] = feed_forward_several(NN, trainx);
        %prediction_train = a{end}';
    end

    repr = a{end-1}';

    save('NN.mat', 'NN');
    
end
