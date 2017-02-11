% The main file

if ((exist('datax') ~= 1) || (exist('datay') ~= 1))
    [datax, datay] = load_data();
end

n = size(datax,1);
testsize = 1000;
nb_tests = 100;

[train_i, test_i] = train_test_sets(n, testsize, nb_tests);

scores = zeros(2, nb_tests);
clear auc14;
clear auc15;

for i=1:size(train_i,2)
    %tt = time();
    trainx = datax(train_i{i},2:end);
    trainy = datay(train_i{i},2:end);
    testx = datax(test_i{i},2:end);
    testy = datay(test_i{i},2:end);
    %time() - tt
    
    % Main part, where all the prediction is done
    %tt = time();
    prediction_train = randi(2,n-testsize,2)-1;
    prediction_test = randi(2,testsize,2)-1;
    %time() - tt
    % End of the main part, here we only compute the error and plot it
    %tt = time();
    [scores(1,i), auc14{1,i}, auc15{1,i}] = compute_auc(prediction_train, trainy);
    [scores(2,i), auc14{2,i}, auc15{2,i}] = compute_auc(prediction_test, testy);
    %time() - tt
end

mean(scores, 2)
plot_auc(auc14, auc15);
