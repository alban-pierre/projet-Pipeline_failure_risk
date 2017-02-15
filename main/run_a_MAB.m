% Runs a multi armed bandit setup
    
% Initialization of paths
init;

% Loading datasets and formatting the data
if ((exist('datainitx') ~= 1) || (exist('datainity') ~= 1))
    [datainitx, datainity] = load_data();
end
if (submit_file && (exist('datas') ~= 1))
    datas = load_data(1);
end

% Options (useless if we are submitting a file)

tt = time();

MAB = create_a_MAB(datainitx, datainity);

tmax = 10;

% Beginning of the UCB algorithm
NbArms=MAB.nbArms;

tmax = max(tmax, NbArms);

mu = zeros(1,NbArms);
smu = zeros(1,NbArms);
na = ones(1,NbArms);
rew = zeros(1,tmax);
rews = zeros(1,NbArms);

for i=1:NbArms
    mu(1,i) = sample_a_MAB(MAB,i);
    MAB.draws(1,i) = MAB.draws(1,i)+1;
    rews(1,i) = rews(1,i) + mu(1,i);
    fprintf(2,'*');
end
fprintf(2,'\n');
rew(1,1:NbArms) = mu;
smu = mu;

rmin = min(rew,[],2);
rmax = max(rew,[],2);

mu = ((smu./na) - rmin)./(rmax - rmin);

for t=NbArms+1:tmax
    [ma, ima] = max(mu+sqrt(log(t)./(2*na)), [], 2);
    rew(1,t) = sample_a_MAB(MAB,ima);
    MAB.draws(1,ima) = MAB.draws(1,ima)+1;
    smu(1,ima) = smu(1,ima) + rew(1,t);
    rews(1,ima) = rews(1,ima) + rew(1,t);
    na(1,ima) = na(1,ima) + 1;
    if (rew(1,t) > rmax)
        rmax = rew(1,t);
        mu = ((smu./na) - rmin)./(rmax - rmin);
    elseif (rew(1,t) < rmin)
        rmin = rew(1,t);
        mu = ((smu./na) - rmin)./(rmax - rmin);
    else
        mu(1,ima) = ((smu(1,ima)/na(1,ima)) - rmin)/(rmax - rmin);
    end
    fprintf(2,'*');
end
draws = na;
%rews = rews./draws;
% End of the UCB algorithm

fprintf(2, 'The MAB algorithm took %f seconds\n', time() - tt);


figure;
plot(1:NbArms, rews./draws, 'r');
hold on;
plot(1:NbArms, draws/max(draws,[],2), 'k');

rews./draws
