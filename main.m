% The main file

submit_file = 0; % Set it to 1 to produce a result file to submit on data challenge, 0 otherwise

% Loading datasets and formatting the data
if ((exist('datainitx') ~= 1) || (exist('datainity') ~= 1))
    [datainitx, datainity] = load_data();
end
if (submit_file && (exist('datas') ~= 1))
    datas = load_data(1);
end

% Options (useless if we are submitting a file)
n = size(datainitx,1);
testsize = 6476; % The testsize
trainsize = n-testsize;
nb_tests = 10; % The number of tests

% Defining train and test sets
[train_i, test_i] = random_train_test_sets(trainsize, testsize, nb_tests);

scores = zeros(2, nb_tests);
clear auc14;
clear auc15;

% Modification of the data representation
tt = time();
datax = datainitx(:,2:end);
datay = datainity(:,2:end);
datax = remove_constant_columns(add_power2_columns(datax, ones(size(datax,2))));

fprintf(2, 'The data representation transformation took %f seconds\n', time() - tt);

if (submit_file) % In the particular case of submitting
    testsize = 9713;
    nb_tests = 1;
    trainx = datainitx(:,2:end);
    trainy = datainity(:,2:end);
    testx = datas(:,2:end);
    train_i = 0;
    scores = 0;
end

% For each train set
tt = time();
for i=1:size(train_i,2)

    % create train and test sets
    if (~submit_file)
        trainx = datax(train_i{i},:);
        trainy = datay(train_i{i},:);
        testx = datax(test_i{i},:);
        testy = datay(test_i{i},:);
    end
    
    % Main part, where all the prediction is done
    if (0)
        prediction_train = randi(2,n-testsize,2)-1;
        prediction_test = randi(2,testsize,2)-1;
    end
    if (1)
        [prediction_train, prediction_test] = ridge_regression(trainx, trainy, testx, 0.1);
    end
    if (0)
        kernel = @(x1, x2) laplacian_kernel(x1, x2, 0.1);
        [prediction_train, prediction_test] = kernel_ridge_regression(kernel, trainx, trainy, testx, 0.1);
    end
    
    % End of the main part, here we only compute the error and plot it

    % Compute scores and curves and store the submission file
    [sc, auc] = auc_error(prediction_train, trainy);
    scores(1,i) = sc*[0.6; 0.4];
    auc14{1,i} = auc{1};
    auc15{1,i} = auc{2};
        
    if (submit_file)
        create_submit_file(prediction_test);
    else
        [sc, auc] = auc_error(prediction_test, testy);
        scores(2,i) = sc*[0.6; 0.4];
        auc14{2,i} = auc{1};
        auc15{2,i} = auc{2};
    end
end
fprintf(2, 'The predictions took %f seconds\n', time() - tt);

% Print scores and plot auc curves
mean(scores, 2)
plot_auc(auc14, -1, {'Training AUC for 2014', 'Testing AUC for 2014'});
plot_auc(auc15, -1, {'Training AUC for 2015', 'Testing AUC for 2015'});
