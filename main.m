% The main file

submit_file = 0; % Set it to 1 to produce a result file to submit on data challenge, 0 otherwise

% Loading datasets and formatting the data
if ((exist('datax') ~= 1) || (exist('datay') ~= 1))
    [datax, datay] = load_data();
end
if (submit_file && (exist('datas') ~= 1))
    datas = load_data(1);
end

% Options (useless if we are submitting a file)
n = size(datax,1);
testsize = 500;%6476; % The testsize
trainsize = 1500;%n-testsize;
nb_tests = 2; % The number of tests

% Defining train and test sets
[train_i, test_i] = random_train_test_sets(trainsize, testsize, nb_tests);

scores = zeros(2, nb_tests);
clear auc14;
clear auc15;

if (submit_file) % In the particular case of submitting
    testsize = 9713;
    nb_tests = 1;
    trainx = datax(:,2:end);
    trainy = datay(:,2:end);
    testx = datas(:,2:end);
    train_i = 0;
    scores = 0;
end

% For each train set
for i=1:size(train_i,2)

    % create train and test sets
    if (~submit_file)
        trainx = datax(train_i{i},2:end);
        trainy = datay(train_i{i},2:end);
        testx = datax(test_i{i},2:end);
        testy = datay(test_i{i},2:end);
    end
    
    % Main part, where all the prediction is done
    if (0)
        prediction_train = randi(2,n-testsize,2)-1;
        prediction_test = randi(2,testsize,2)-1;
    end
    if (0)
        [prediction_train, prediction_test] = ridge_regression(trainx, trainy, testx, 0.1);
    end
    if (1)
        kernel = @(x1, x2) laplacian_kernel(x1, x2, 0.1);
        [prediction_train, prediction_test] = kernel_ridge_regression(kernel, trainx, trainy, testx, 0.1);
    end
    
    % End of the main part, here we only compute the error and plot it

    % Compute scores and curves and store the submission file
    [scores(1,i), auc14{1,i}, auc15{1,i}] = compute_auc(prediction_train, trainy);
        
    if (submit_file)
        create_submit_file(prediction_test);
    else
        [scores(2,i), auc14{2,i}, auc15{2,i}] = compute_auc(prediction_test, testy);
    end
end

% Print scores and plot auc curves
mean(scores, 2)
plot_auc(auc14, auc15);
