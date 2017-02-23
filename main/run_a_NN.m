% The main file

% Initialization of paths
init;
submit_file = 0; % Set it to 1 to produce a result file to submit on data challenge, 0 otherwise

% Loading datasets and formatting the data
if ((exist('datainitx') ~= 1) || (exist('datainity') ~= 1))
    [datainitx, datainity] = load_data();
end
if (submit_file && (exist('datas') ~= 1))
    datas = load_data(1);
end

% Options (useless if we are submitting a file)
trainsize = 12951; % The trainsize
testsize = 6476; % The testsize
nb_tests = 1; % The number of tests
setrand = 1; % The random generator beginning (-1 = no set)
k = 10; %k of k_fold sets
algo = algo_options();

% Defining train and test sets
if (1)
    [train_i, test_i] = random_train_test_sets(trainsize, testsize, nb_tests, setrand);
elseif (0)
    [train_i, test_i] = k_fold_train_test_sets(size(datainitx,1), k, nb_tests, setrand)
end
scores = zeros(2, size(train_i, 2));
clear auc14;
clear auc15;

% Modification of the data representation
tt = time();
datax = datainitx(:,2:end);
datay = datainity(:,2:end);
%datax = remove_constant_columns(add_power2_columns(datax, ones(size(datax,2))));
datax = set_fixed_mean(datax);
datax = set_fixed_variance(datax);
%datax = remove_constant_columns(add_power2_columns(datax, ones(size(datax,2))));


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
    if (submit_file)
        [prediction_test, err_train, auc_train] = prediction(algo, trainx, trainy, testx);
        create_submit_file(prediction_test);
    else
        % create train and test sets
        trainx = datax(train_i{i},:);
        trainy = datay(train_i{i},:);
        testx = datax(test_i{i},:);
        testy = datay(test_i{i},:);
        % Prediction
        %[err_train, err_test, auc_train, auc_test] = prediction_error(algo, trainx, trainy, testx, testy);
        NN = create_a_NN(algo);
        NNN = NN;
        N = size(trainx,1);
        r = randperm(N);
        clear err_tr;
        clear err_te;
        
        for kk = 1:25
            r = randperm(N);
            NN = train_a_NN(NN, algo, trainx(r,:), trainy(r,:), 1);
            [NN, a] = feed_forward_several(NN, trainx);
            prediction_train = a{end}';
            [NN, a] = feed_forward_several(NN, testx);
            prediction_test = a{end}';
            [err_tr(kk,:), ~] = auc_error(prediction_train, trainy);
            [err_te(kk,:), ~] = auc_error(prediction_test, testy);
            fprintf(2,'*');
        end
        fprintf(2,'\n');
        
        [NN, a] = feed_forward_several(NN, trainx);
        prediction_train = a{end}';
        [NN, a] = feed_forward_several(NN, testx);
        prediction_test = a{end}';
            
        [err_train, auc_train] = auc_error(prediction_train, trainy);
        [err_test, auc_test] = auc_error(prediction_test, testy);

        
        scores(2,i) = err_test*[0.6; 0.4];
        auc14{2,i} = auc_test{1};
        auc15{2,i} = auc_test{2};
    end
    scores(1,i) = err_train*[0.6; 0.4];
    auc14{1,i} = auc_train{1};
    auc15{1,i} = auc_train{2};
end
fprintf(2, 'The predictions took %f seconds\n', time() - tt);

% Print scores and plot auc curves
mean(scores, 2)
plot_auc(auc14, -1, {'Training AUC for 2014', 'Testing AUC for 2014'});
plot_auc(auc15, -1, {'Training AUC for 2015', 'Testing AUC for 2015'});

figure;
hold on;
plot(1:size(err_tr,1), err_tr(:,1)', 'k-');
plot(1:size(err_tr,1), err_tr(:,2)', 'r-');
plot(1:size(err_tr,1), err_te(:,1)', 'k.');
plot(1:size(err_tr,1), err_te(:,2)', 'r.');
