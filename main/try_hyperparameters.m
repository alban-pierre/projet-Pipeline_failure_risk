% The main file

% Initialization of paths
init;

% Loading datasets and formatting the data
if ((exist('datainitx') ~= 1) || (exist('datainity') ~= 1))
    [datainitx, datainity] = load_data();
end

% Options (useless if we are submitting a file)
trainsize = 6476;%12951; % The trainsize
testsize = 12951;%6476; % The testsize
nb_tests = 100; % The number of tests
setrand = 3; % The random generator beginning (-1 = no set)
k = 10; %k of k_fold sets
algo = algo_options();
kernel_hyp = 1:1;
regr_hyp = 10.^(3:7);


% Defining train and test sets
if (1)
    [train_i, test_i] = random_train_test_sets(trainsize, testsize, nb_tests, setrand);
elseif (0)
    [train_i, test_i] = k_fold_train_test_sets(size(datainitx,1), k, nb_tests, setrand)
end
err_train = zeros(size(kernel_hyp,2), size(regr_hyp,2));
err_test = zeros(size(kernel_hyp,2), size(regr_hyp,2));

% Modification of the data representation
tt = time();
datax = datainitx(:,2:end);
datay = datainity(:,2:end);
datax = remove_constant_columns(add_power2_columns(datax, ones(size(datax,2))));

fprintf(2, 'The data representation transformation took %f seconds\n', time() - tt);

% For each train set
tt = time();
for i=1:size(train_i,2)
    % create train and test sets
    trainx = datax(train_i{i},:);
    trainy = datay(train_i{i},:);
    testx = datax(test_i{i},:);
    testy = datay(test_i{i},:);
    % Prediction
	for j = 1:size(kernel_hyp,2)
		for k = 1:size(regr_hyp,2)
			algo.kernel_hyp = kernel_hyp(1,j);
			algo.regr_hyp = regr_hyp(1,k);
			[err_tr, err_te, ~, ~] = prediction_error(algo, trainx, trainy, testx, testy);
			err_train(j,k) = err_train(j,k) + err_tr*[0.6; 0.4];
			err_test(j,k) = err_test(j,k) + err_te*[0.6; 0.4];
		end
	end
end
fprintf(2, 'The predictions took %f seconds\n', time() - tt);

err_train = err_train/nb_tests;
err_test = err_test/nb_tests;

figure;
plot(1:size(regr_hyp,2), err_train, 'k');
hold on;
plot(1:size(regr_hyp,2), err_test, 'r');
