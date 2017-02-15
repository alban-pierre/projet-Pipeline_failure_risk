function rew = sample_a_MAB(MAB, j)

    % Function that samples a MAB

    assert(j <= size(MAB.arm, 2), 'Error : you cannot sample an inexisting arm');
    
    [train_i, test_i] = random_train_test_sets(MAB.trainsize, MAB.testsize, 1, MAB.draws(1,j)+1);

    trainx = MAB.datax{MAB.data{j}};
    trainx = trainx(train_i{1},:);
    trainy = MAB.datay{MAB.data{j}};
    trainy = trainy(train_i{1},:);
    testx = MAB.datax{MAB.data{j}};
    testx = testx(test_i{1},:);
    testy = MAB.datay{MAB.data{j}};
    testy = testy(test_i{1},:);
    % Prediction
    [err_train, err_test, ~, ~] = prediction_error(MAB.arm{j}, trainx, trainy, testx, testy);
    rew = err_test*[0.6; 0.4];
    
end
        
