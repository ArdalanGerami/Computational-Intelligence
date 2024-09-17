%% part a, b
clear all; clc; close all;

data = load('Project_data.mat');
TrainData = data.TrainData;
TrainLabel = data.TrainLabels;
TestData = data.TestData;
[numberOfChannels, numberOfSamples, numberOfTrials] = size(TrainData);
Fs = data.fs;
N1 = length(find(TrainLabel == 1));
N2 = length(find(TrainLabel == -1));
index1 = find(TrainLabel == 1);
index_1 = find(TrainLabel == -1);

% Average Feature :
for channel = 1: numberOfChannels
    A = TrainData(channel, :, :);
    A = transpose(squeeze(A));
    for i = 1: numberOfTrials
        average(channel, i) = sum(A(i, :)) / numberOfSamples;
    end
    average_J(channel) = fisher(average,channel,index1,index_1);
end
average = normalize(average);
disp(['Fisher score for best average is ', num2str(max(average_J)), ' for channel ', num2str(find(average_J == max(average_J)))]);
J_mat = average_J; 
% Variance Feature :
for channel = 1: numberOfChannels
    A = TrainData(channel, :, :);
    A = transpose(squeeze(A));
    for i = 1:numberOfTrials
        variance(channel, i) = var(A(i, :));
    end
    variance_J(channel) = fisher(variance,channel,index1,index_1);
end
variance = normalize(variance);
disp(['Fisher score for best variance is ', num2str(max(variance_J)), ' for channel ', num2str(find(variance_J == max(variance_J)))]);
J_mat = [J_mat; variance_J];

%Skewness Feature:
for channel = 1: numberOfChannels
    A = TrainData(channel, :, :);
    A = transpose(squeeze(A));
    for i = 1:numberOfTrials
        a_skewness(channel, i) = skewness(A(i, :));
    end
    skewness_J(channel) = fisher(a_skewness,channel,index1,index_1);
end
a_skewness = normalize(a_skewness);
disp(['Fisher score for best skewness is ', num2str(max(skewness_J)), ' for channel ', num2str(find(skewness_J == max(skewness_J)))]);
J_mat = [J_mat; skewness_J];

%Entropy Feature:
for channel = 1:numberOfChannels
    A = TrainData(channel, :, :);
    A = transpose(squeeze(A));
    for i = 1:numberOfTrials
        Entropy(channel, i) = entropy(A(i, :));
    end
    entropy_J(channel) = fisher(Entropy,channel,index1,index_1);
end
Entropy = normalize(Entropy);
disp(['Fisher score for best entropy is ', num2str(max(entropy_J)), ' for channel ', num2str(find(entropy_J == max(entropy_J)))]);
J_mat = [J_mat; entropy_J];

%Medium Frequency Feature:
for channel = 1: numberOfChannels
    A = TrainData(channel, :, :);
    A = transpose(squeeze(A));
    for i = 1: numberOfTrials
        mf(channel, i) = medfreq (A(i, :));
    end
    mfj(channel) = fisher(mf,channel,index1,index_1); 
end
mf = normalize(mf);
disp(['Fisher score for best medium frequency is ', num2str(max(mfj)), ' for channel ', num2str(find(mfj == max(mfj)))]);
J_mat = [J_mat; mfj];

% Mean Frequency Feature :
for channel = 1:numberOfChannels
    A = TrainData(channel, :, :);
    A = transpose(squeeze(A));
    for i = 1: numberOfTrials
        meanFrequency(channel, i) = meanfreq(A(i, :));
    end
    meanFrequency_J(channel) = fisher(meanFrequency,channel,index1,index_1);
end
meanFrequency = normalize(meanFrequency);
disp(['Fisher score for best mean frequency is ', num2str(max(meanFrequency_J)), ' for channel ', num2str(find(meanFrequency_J == max(meanFrequency_J)))]);
J_mat = [J_mat; meanFrequency_J];

% Band Power Feature:
for channel = 1:numberOfChannels
    A = TrainData(channel, :, :);
    A = transpose(squeeze(A));
    for i = 1: numberOfTrials
        bp(channel, i) = bandpower(A(i, :));
    end
    bpJ(channel) = fisher(bp,channel,index1,index_1);
end
bp = normalize(bp);
disp(['Fisher score for best band power is ', num2str(max(bpJ)), ' for channel ', num2str(find(bpJ == max(bpJ)))]);
J_mat = [J_mat; bpJ];

% 99 percent Bandwidth:

for channel = 1: numberOfChannels
    A = TrainData(channel, :, :);
    A = transpose(squeeze(A));
    for i = 1: numberOfTrials
        b99p(channel, i) = obw(A(i, :));
    end
    b99p_J(channel) = fisher(b99p,channel,index1,index_1); 
end
b99p = normalize(b99p);
disp(['Fisher score for best 99 percent bandwidth = ', num2str(max(b99p_J)), ' for channel = ', num2str(find(b99p_J == max(b99p_J)))]);
J_mat = [J_mat; b99p_J];

% Maximum Power Frequency:
for channel = 1: numberOfChannels
    A = TrainData(channel, :, :);
    A = transpose(squeeze(A));
    for i = 1: numberOfTrials
        x = A(i, :);
        n = length(x);
        y = fftshift(fft(x));
        f = (-n/2:n/2-1)*(Fs/n);       % 0-centered frequency range
        power = abs(y).^2/n;           % 0-centered power
        index = find(power == max(power));
        mpf(channel, i) = index(end);
    end
    mpf_J(channel) = fisher(mpf,channel,index1,index_1);
end
mpf = normalize(mpf);
disp(['Fisher score for best maximum power frequency is ', num2str(max(mpf_J)), ' for channel ', num2str(find(mpf_J == max(mpf_J)))]);
J_mat = [J_mat; mpf_J];

%% Best Channel for each feature

disp('--------------------------------------------------------------------------------------------------------------------------------------------------');
for i = 1:numberOfChannels
    temp(i) = find(J_mat(:, i) == max(J_mat(:, i)));
    disp(['The best feature for channel ', num2str(i), ' is ', featurestring(temp(i)), ', also its fisher score is ', num2str(max(J_mat(:, i)))]);
end


% Best group of features :
disp('              best group of features             '); 
features1 = transpose([mf(43, :); meanFrequency(23, :); mpf(56, :)]);
features2 = transpose([mf(43, :); meanFrequency(23, :); a_skewness(11, :)]);
features3 = transpose([mf(43, :); meanFrequency(23, :); average(47,:)]);
features4 = transpose([mf(43, :);mf(59,:);  meanFrequency(23, :); mpf(24, :); a_skewness(25, :); a_skewness(49, :);average(47,:)]);
features5 = transpose([mf(43, :);mf(59,:);mf(33,:);meanFrequency(23, :); mpf(24, :) ; a_skewness(11, :); a_skewness(25, :); average(47,:)]);
features6 = transpose([mf(43, :);mf(59,:);mf(33,:); meanFrequency(23, :); mpf(24, :);mpf(56, :); a_skewness(11, :); a_skewness(25, :); a_skewness(49, :);average(47,:)]);

%features1
featureMatrix = features1;
J = groupfisher( featureMatrix,index1,index_1,N1,N2 );
disp('--------------------------------------------------------------------------------------------------------------------------------------------------');
disp(['for feature group 1 ---> J = ', num2str(J)]);
featureMatrix = featureMatrix';

%features2
featureMatrix = features2;
J = groupfisher( featureMatrix,index1,index_1,N1,N2 );
disp('--------------------------------------------------------------------------------------------------------------------------------------------------');
disp(['for feature group 2 ---> J = ', num2str(J)]);
featureMatrix = featureMatrix';

%features3
featureMatrix = features3;
J = groupfisher( featureMatrix,index1,index_1,N1,N2 );
disp('--------------------------------------------------------------------------------------------------------------------------------------------------');
disp(['for feature group 3 ---> J = ', num2str(J)]);
featureMatrix = featureMatrix';

%features4
featureMatrix = features4;
J = groupfisher( featureMatrix,index1,index_1,N1,N2 );
disp('--------------------------------------------------------------------------------------------------------------------------------------------------');
disp(['for feature group 4 ---> J = ', num2str(J)]);
featureMatrix = featureMatrix';

%features5
featureMatrix = features5;
J = groupfisher( featureMatrix,index1,index_1,N1,N2 );
disp('--------------------------------------------------------------------------------------------------------------------------------------------------');
disp(['for feature group 5 ---> J = ', num2str(J)]);
featureMatrix = featureMatrix';

%features6
featureMatrix = features6;
J = groupfisher( featureMatrix,index1,index_1,N1,N2 );
disp('--------------------------------------------------------------------------------------------------------------------------------------------------');
disp(['for feature group 6 ---> J = ', num2str(J)]);
featureMatrix = featureMatrix';

%% part c

[numberOfChannelsT, numberOfSamplesT, numberOfTrialsT] = size(TestData);

for channel = 1:numberOfChannelsT
    A = TestData(channel, :, :);
    A = transpose(squeeze(A));
    for i = [1: 1: numberOfTrialsT]
        test_average(channel, i) = sum(A(i, :)) / numberOfSamples;
        test_variance(channel, i) = var(A(i, :));
        test_Skewness(channel, i) = skewness(A(i, :));
        test_Entropy(channel, i) = entropy(A(i, :));
        test_mediumFrequency(channel, i) = medfreq (A(i, :));
        test_meanFrequency(channel, i) = meanfreq(A(i, :));
        test_Bandpower(channel, i) = bandpower(A(i, :));
        bandwidth_99PercentT(channel, i) = obw(A(i, :));
        
        xT = A(i, :);
        nT = length(xT);
        yT = fftshift(fft(xT));
        fT = (-nT/2:nT/2-1)*(Fs/n);       % 0-centered frequency range
        powerT = abs(yT).^2/nT;           % 0-centered power
        indexT = find(powerT == max(powerT));
        test_mpf(channel, i) = indexT(end);
    end
end

test_average = normalize(test_average);
test_variance = normalize(test_variance);
test_Skewness = normalize(test_Skewness);
test_Entropy = normalize(test_Entropy);
test_mediumFrequency = normalize(test_mediumFrequency);
test_meanFrequency = normalize(test_meanFrequency);
test_Bandpower = normalize(test_Bandpower);


featureMatrixT = transpose(features6);
% MLP:
activation_functions = ["radbas", "logsig", "purelin", "satlin", "tansig", "hardlims"];
betsACC = -inf;

for i = 1:length(activation_functions)
    for N=1:20

        ACC = 0 ;
        % 5-fold cross-validation
        for k=1:5
            train_indices = [1:(k-1)*110,k*110+1:550] ;
            valid_indices = (k-1)*110+1:k*110;

            TrainX = featureMatrix(:,train_indices) ;
            ValX = featureMatrix(:,valid_indices) ;
            TrainY = TrainLabel(train_indices) ;
            ValY = TrainLabel(valid_indices) ;

            net = patternnet(N);
            net = train(net,TrainX,TrainY);
            
            activation_function_string = convertStringsToChars(activation_functions(i));
            net1.layers{2}.transferFcn =  activation_function_string;

            predict_y = net(ValX);
            Thr = 0.5 ;

            predict_y = predict_y >= Thr ;

            ACC = ACC + length(find(predict_y==ValY)) ;
        end

        if (ACC > betsACC)
            bestACC = ACC;
            predictTest(1, :) = net(featureMatrixT);
            predictTest(1, :) = predictTest(1, :) >= Thr ;
            phase1mlp = predictTest(1, :);
        end
        ACCMat(N) = ACC/440 ;
    end
    best_N = find(ACCMat == max(ACCMat));
    disp(['Best Number of Neurons in MLP with actvation function ', activation_function_string, ' is ', num2str(best_N), ', and Accuracy = ', num2str(ACCMat(best_N))]);
end
save('phase1mlp.mat');

%% RBF 
goal = 0;
k = [1: 1 :10];
mn = [1: 1: 40];
ACCMat = [];
betsACC = -inf;
maximumNumberOfHiddenNeurons = [];
for spread = k
    for MN = mn
         ACC = 0;
            for k=1:5
                train_indices = [1:(k-1)*110,k*110+1:550];
                valid_indices = (k-1)*110+1:k*110;
                TrainX = featureMatrix(:,train_indices);
                ValX = featureMatrix(:,valid_indices);
                TrainY = TrainLabel(train_indices);
                ValY = TrainLabel(valid_indices);
                
                evalc('net = newrb(TrainX, TrainY, goal, spread, MN)');
                predict_y = sim(net, ValX);
                
                
                Thr = 0.5 ;
                predict_y = predict_y >= Thr;
                ACC = ACC + length(find(predict_y==ValY)) ;               
            end
            if (ACC > betsACC)
                bestACC = ACC;
                predictTest(2, :) = sim(net, featureMatrixT);
                predictTest(2, :) = predictTest(2, :) >= Thr ;
                phase1rbf=predictTest(2, :);
            end
         ACCMat(spread, MN) = ACC/150 ;
    end
end

maximum = max(max(ACCMat));
[numberOfHiddenNeurons, Radius] = find(ACCMat == maximum);

disp(['Best Number of Neurons in RBF is ', num2str(numberOfHiddenNeurons(1)), ', Radius is ', num2str(Radius(1)) ' and Accuracy is ', num2str(maximum(1))]);

save('phase1rbf.mat');
%% Phase 2

J_mat_reshaped = reshape(J_mat, [1, 59*9]);
J_vector_sorted = sort(J_mat_reshaped);
top_50_FisherVals = J_vector_sorted(end - 58:end);
for i = 1: 59
    [feature(i), channel(i)] = find(J_mat == top_50_FisherVals(i));
end

for i = [1: 1: 59]
    activation_function_string = num2feature(feature(i));
    
    if (activation_function_string == "Average")
        BestFeatureMat_GA(i, :) = average(channel(i), :); 
    end
    
    if (activation_function_string == "Variance")
        BestFeatureMat_GA(i, :) = variance(channel(i), :); 
    end
    
    if (activation_function_string == "Skewness")
        BestFeatureMat_GA(i, :) = a_skewness(channel(i), :); 
    end
    
    if (activation_function_string == "Entropy")
        BestFeatureMat_GA(i, :) = Entropy(channel(i), :); 
    end
    
    if (activation_function_string == "Medium Frequency")
        BestFeatureMat_GA(i, :) = mf(channel(i), :); 
    end
    
    if (activation_function_string == "Mean Frequency")
        BestFeatureMat_GA(i, :) = meanFrequency(channel(i), :); 
    end
    
    if (activation_function_string == "Band Power")
        BestFeatureMat_GA(i, :) = bp(channel(i), :); 
    end
    
    if (activation_function_string == "99 percent Bandwidth")
        BestFeatureMat_GA(i, :) = b99p(channel(i), :); 
    end
    
    if (activation_function_string == "Maximum Power Frequency")
        BestFeatureMat_GA(i, :) = mpf(channel(i), :); 
    end
     
end

save('BestFeatureMat_GA.mat');

%% Selected features for test data  

selecttive_features = [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];

j = 1;
for i = 1:59
    if (selecttive_features(i) == 1)
        activation_function_string = featurestring(feature(i));
    
        if (activation_function_string == "Average")
            disp([num2str(j), ': Average Feature for channel ', num2str(channel(i))]);
        end

        if (activation_function_string == "Variance")
            disp([num2str(j), ': Variance Feature for channel ', num2str(channel(i))]);
        end

        if (activation_function_string == "Skewness")
            disp([num2str(j), ': Skewness Feature for channel ', num2str(channel(i))]); 
        end

        if (activation_function_string == "Entropy")
            disp([num2str(j), ': Entropy Feature for channel ', num2str(channel(i))]);
        end

        if (activation_function_string == "Medium Frequency")
            disp([num2str(j), ': Medium Frequency Feature for channel ', num2str(channel(i))]);
        end

        if (activation_function_string == "Mean Frequency")
            disp([num2str(j), ': Mean Frequency Feature for channel ', num2str(channel(i))]); 
        end

        if (activation_function_string == "Band Power")
            disp([num2str(j), ': Band Power Feature for channel ', num2str(channel(i))]); 
        end

        if (activation_function_string == "99 percent Bandwidth")
            disp([num2str(j), ': 99 percent Bandwidth Feature for channel ', num2str(channel(i))]);
        end

        if (activation_function_string == "Maximum Power Frequency")
            disp([num2str(j), ': Maximum Power Frequency Feature for channel ', num2str(channel(i))]);
        end
        j = j + 1;
    end
    
end

selective_indexes = find(selecttive_features == 1);

best_features_GA_Train = BestFeatureMat_GA(selective_indexes, :);


for i = [1: 1: 30]
    activation_function_string = num2feature(feature(i));
    
    if (activation_function_string == "Average")
        BestFeatureMat_GA_Test(i, :) = test_average(channel(i), :); 
    end
    
    if (activation_function_string == "Variance")
        BestFeatureMat_GA_Test(i, :) = test_variance(channel(i), :); 
    end
    
    if (activation_function_string == "Skewness")
        BestFeatureMat_GA_Test(i, :) = test_Skewness(channel(i), :); 
    end
    
    if (activation_function_string == "Entropy")
        BestFeatureMat_GA_Test(i, :) = test_Entropy(channel(i), :); 
    end
    
    if (activation_function_string == "Medium Frequency")
        BestFeatureMat_GA_Test(i, :) = test_mediumFrequency(channel(i), :); 
    end
    
    if (activation_function_string == "Mean Frequency")
        BestFeatureMat_GA_Test(i, :) = test_meanFrequency(channel(i), :); 
    end
    
    if (activation_function_string == "Band Power")
        BestFeatureMat_GA_Test(i, :) = test_Bandpower(channel(i), :); 
    end
    
    if (activation_function_string == "99 percent Bandwidth")
        BestFeatureMat_GA_Test(i, :) = bandwidth_99PercentT(channel(i), :); 
    end
    
    if (activation_function_string == "Maximum Power Frequency")
        BestFeatureMat_GA_Test(i, :) = test_mpf(channel(i), :); 
    end
     
end

best_features_GA_Test = BestFeatureMat_GA_Test(selective_indexes, :);

%% MLP 

clc;
activation_functions = ["radbas", "logsig", "purelin", "satlin", "tansig", "hardlims"];
ACCMat = [];
betsACC = -inf;
for i = 1:length(activation_functions)
    for N=1:20

        ACC = 0 ;
        % 6-fold cross-validation
        for k=1:5
            train_indices = [1:(k-1)*110,k*110+1:550] ;
            valid_indices = (k-1)*110+1:k*110;

            TrainX = best_features_GA_Train(:,train_indices) ;
            ValX = best_features_GA_Train(:,valid_indices) ;
            TrainY = TrainLabel(train_indices) ;
            ValY = TrainLabel(valid_indices) ;

            net = patternnet(N);
            net = train(net,TrainX,TrainY);
            
            activation_function_string = convertStringsToChars(activation_functions(i));
            net1.layers{2}.transferFcn =  activation_function_string;

            predict_y = net(ValX);
            Thr = 0.5 ;

            predict_y = predict_y >= Thr ;

            ACC = ACC + length(find(predict_y==ValY)) ;
        end

        if (ACC > betsACC)
            bestACC = ACC;
            predictTest(3, :) = net(best_features_GA_Test);
            predictTest(3, :) = predictTest(3, :) >= Thr ;
            phase2mlp = predictTest(3, :)
        end
        ACCMat(N) = ACC/450 ;
    end
    best_N = find(ACCMat == max(ACCMat));
    %disp(['Best Number of Neurons in MLP = ', num2str(best_N(1)), ', Accuracy = ', num2str(bestACC / 165)]);
    disp(['Best Number of Neurons in MLP with actvation function ', activation_function_string, ' is equal to ', num2str(best_N), ', Accuracy = ', num2str(ACCMat(best_N))]);
end

%% RBF 

goal = 0;
k = [1: 1 :10];
mn = [1: 1: 40];
ACCMat = [];
betsACC = -inf;
maximumNumberOfHiddenNeurons = [];
for spread = k
    for MN = mn
         ACC = 0;
            for k=1:5
                train_indices = [1:(k-1)*110,k*110+1:550];
                valid_indices = (k-1)*110+1:k*110;
                TrainX = best_features_GA_Train(:,train_indices);
                ValX = best_features_GA_Train(:,valid_indices);
                TrainY = TrainLabel(train_indices);
                ValY = TrainLabel(valid_indices);
                
                evalc('net = newrb(TrainX, TrainY, goal, spread, MN)');
                predict_y = sim(net, ValX);
                
                
                Thr = 0.5 ;
                predict_y = predict_y >= Thr;
                ACC = ACC + length(find(predict_y==ValY)) ;               
            end
            if (ACC > betsACC)
                bestACC = ACC;
                predictTest(4, :) = sim(net, best_features_GA_Test);
                predictTest(4, :) = predictTest(4, :) >= Thr ;
                phase2rbf=predictTest(4, :);
            end
         ACCMat(spread, MN) = ACC/165 ;
    end
end

maximum = max(max(ACCMat));
[numberOfHiddenNeurons, Radius] = find(ACCMat == maximum);

disp(['Best Number of Neurons in RBF = ', num2str(numberOfHiddenNeurons(1)), ', Radius = ', num2str(Radius(1)) ', Accuracy = ', num2str(maximum(1))]);

%% Saving Results 

save('phase2mlp.mat');
bestLabels = predictTest(2, :);
save('phase2rbf.mat');

%}
function [e] = fisher(input,ch,index1,index_1)
    mu0 = sum(input(ch, :)) / length(input);
    mu1 = sum(input(ch, index1)) / length(index1);
    mu2 = sum(input(ch, index_1)) / length(index_1);
    var1 = var(input(ch, index1));
    var2 = var(input(ch, index_1));
    e = ((mu0 - mu1)^2 + (mu0 - mu2)^2) / (var1 + var2); 
end

function [J] = groupfisher( features,index1,index_1,N1,N2 )
mu1 = sum(features(index1, :)) / N1;
mu2 = sum(features(index_1, :)) / N2;
mu0 = sum(features) / (N1 + N2);   
[l1 , l2] = size(features);
s1 = zeros(l2);
for i = index1
    s1 = s1 + (features(i, :) - mu1) * transpose(features(i, :) - mu1);
end
s1 = s1 / N1;

s2 = zeros(l2);
for i = index_1
    s2 = s2 + (features(i, :) - mu2) * transpose(features(i, :) - mu2);
end
s2 = s2 / N2;

Sw = s1 + s2;
Sb = (mu1 - mu0) * transpose(mu1 - mu0) + (mu2 - mu0) * transpose(mu2 - mu0);
J = (trace(Sb)/trace(Sw));
end





function [normalized, mean , STD] = normalize(input)
    [l1, l2] = size(input);
    for i = 1: l1
        mean = sum(input(i, :)) / l2;
        STD = std(input(i, :));
        normalized(i, :) = (input(i, :) - mean) / STD;
    end   
end

function [s] = featurestring(input)
    if (input == 1)
        s = 'Average';
    end

    if (input == 2)
        s = 'Variance';
    end

    if (input == 3)
        s = 'Skewness';
    end

    if (input == 4)
        s = 'Entropy';
    end

    if (input == 5)
        s = 'Medfreq';
    end

    if (input == 6)
        s = 'Meanfreq';
    end

    if (input == 7)
        s = 'Bandpower';
    end

    if (input == 8)
        s = 'Occupied Bandwidth';
    end

    if (input == 9)
        s = 'Maxpowerfreq';
    end
    %{
    if (input == 10)
        s = 'AR';
    end
    %}
end    


function [normalized] = normalizedT(input , mean , STD)
    [l1, l2] = size(input);
    for i = 1: l1
        normalized(i, :) = (input(i, :) - mean) / STD;
    end   
end

