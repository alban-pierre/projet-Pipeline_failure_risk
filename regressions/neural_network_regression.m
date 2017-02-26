function [prediction_train, prediction_test] = neural_network_regression(deep, trainx, trainy, testx, testy)

    % Prediction based on neural network, trained from scratch

    % Dimensions :
    % Ntr : Number of training examples
    % Nte : Number of testing examples
    % Dx  : Dimension of examples
    % Dy  : Dimension of prediction
    
    % Input :
    % trainx : (Ntr*Dx)    : Training set
    % trainy : (Ntr*Dy)    : Training output
    % testx  : (Nte*Dx)    : Testing set
    % testy  : (Nte*Dy)    : Testing output, used only if we want to print the testing error accross epochs
    % deep   : (structure) : Parameters of the neural network

    % Output :
    % prediction_train : (Ntr*Dy) : The training prediction
    % prediction_test  : (Nte*Dy) : The testing prediction
    
    assert(nargin >= 4, 'Error : neural_network_regression requires at least 4 arguments : NN parameters, trainx, trainy and testx');
    assert((nargin >= 5) || (deep.show_epoch_err == 0), 'Error : neural_network_regression requires at least 5 arguments if you plot the progress : NN parameters, trainx, trainy, testx and testy');
    assert(size(trainx,1) == size(trainy,1), 'trainx and trainy must contain the same number of examples');
    assert(size(trainx,2) == size(testx,2), 'trainx and testx examples must have the same dimension');
    
    [Ntr, Dx] = size(trainx);
    Dy = size(trainy, 2);
    
    algoo.deep = deep;

    NN = create_a_NN(algoo);
    %load('NN7.mat');
    
    if (deep.show_epoch_err ~= 0)
        err_tr = zeros(deep.epoch,Dy);
        err_te = zeros(deep.epoch,Dy);
        for kk = 1:deep.epoch
            r = randperm(Ntr);
            NN = train_a_NN(NN, algoo, trainx(r,:), trainy(r,:), deep.learn_rate(kk));
            [NN, a] = feed_forward_several(NN, trainx);
            prediction_train = a{end}';
            [NN, a] = feed_forward_several(NN, testx);
            prediction_test = a{end}';
            [err_tr(kk,:), ~] = auc_error(prediction_train, trainy);
            [err_te(kk,:), ~] = auc_error(prediction_test, testy);
            fprintf(2,'Ep%d ', kk);
        end
        fprintf(2,'\n');
        
        figure(deep.show_epoch_err); hold on;
        plot(1:size(err_tr,1), err_tr(:,1)', 'k-');
        plot(1:size(err_tr,1), err_tr(:,2)', 'r-');
        plot(1:size(err_tr,1), err_te(:,1)', 'k.');
        plot(1:size(err_tr,1), err_te(:,2)', 'r.');
        legend('train 2014', 'train 2015', 'test 2014', 'test 2015', 'location', 'southeast');
    else
        for kk = 1:deep.epoch
            r = randperm(Ntr);
            NN = train_a_NN(NN, algoo, trainx(r,:), trainy(r,:), deep.learn_rate(kk));
            fprintf(2,'Ep%d ', kk);
        end
        fprintf(2,'\n');
    
        [NN, a] = feed_forward_several(NN, trainx);
        prediction_train = a{end}';
        [NN, a] = feed_forward_several(NN, testx);
        prediction_test = a{end}';  
    end
    
    save('NN.mat', 'NN');
    
end
