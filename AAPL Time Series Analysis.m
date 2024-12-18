%% Import Data
filename = 'C:\\Users\\Ildar Shakirzianov\\Documents\\AAPL.csv';

% Read the Excel file into a table

AAPL = readtable(filename);
AAPL.Return = log(AAPL.AdjustedOpen) - log(AAPL.AdjClose);
AAPL = AAPL(2:end, :);
data = AAPL.Return;
%diff_data = diff(data);

% Plot the histogram
figure('Position', [15, 10, 1500, 1000]);
histogram(data, 30);
%title('Distribution of AAPL Returns');
xlabel('Log Returns');
ylabel('Frequency');
set(gca, 'FontSize', 45);

% Plot the time series
figure('Position', [15, 10, 1500, 1000]);
plot(AAPL.Date, AAPL.Return);
%title('AAPL Log Returns Time Series');
xlabel('Date');
ylabel('Log Returns');
set(gca, 'FontSize', 45);
%% Stationarity of return

% Calculate the ADF test statistic
[h, pValue_adf, ~, ~] = adftest(data);

% Interpretation:
disp(h);
disp(pValue_adf);

if h == 1
    disp('The series is stationary (reject null hypothesis).');
else
    disp('The series is non-stationary (fail to reject null hypothesis).');
end

% Perform the KPSS test
[h_kpss, pValue_kpss, ~, ~] = kpsstest(data);

% Interpretation:
disp(h_kpss);
disp(pValue_kpss);

if h == 1
    disp('The series is non-stationary (reject null hypothesis).');
else
    disp('The series is stationary (fail to reject null hypothesis).');
end

% Perform the Phillips-Perron test 
[h_pp, pValue_pp, ~, ~] = pptest(data);

% Interpretation for Phillips-Perron test:
disp(h_pp);
disp(pValue_pp);

if h_pp == 1
    disp('The series is stationary (reject null hypothesis based on Phillips-Perron test).');
else
    disp('The series is non-stationary (fail to reject null hypothesis based on Phillips-Perron test).');
end
%% Test of normality 
% Perform the Jarque-Bera test
[h_jb, pValue_jb, ~, ~] = jbtest(data);

% Interpretation for Jarque-Bera test:
disp(h_jb)
disp(pValue_jb);

if h_jb
    disp('The data is not normally distributed (reject null hypothesis based on Jarque-Bera test).');
else
    disp('The data follows a normal distribution (fail to reject null hypothesis based on Jarque-Bera test).');
end

% Perform the Anderson-Darling test
[h_ad, pValue_ad, ~, ~] = adtest(data);

% Interpretation for Anderson-Darling test:
disp(h_ad);
disp(pValue_ad);

if h_ad
    disp('The data is not normally distributed (reject null hypothesis based on Anderson-Darling test).');
else
    disp('The data follows a normal distribution (fail to reject null hypothesis based on Anderson-Darling test).');
end

% Perform the Shapiro-Wilk and Shapiro-Francia tests
%[h_sw, pValue_sw] = swtest(data);

% Interpretation for Shapiro-Wilk test:
%disp(h_sw)
%disp(pValue_sw);

%if h_sw
%    disp('The data is not normally distributed (reject null hypothesis based on Shapiro-Wilk test).');
%else
%    disp('The data follows a normal distribution (fail to reject null hypothesis based on Shapiro-Wilk test).');
%end

%% Test of distribution

% Calculate skewness
skewnessValue = skewness(data);
disp(['Skewness = ', num2str(skewnessValue)])

% Calculate kurtosis
kurtosisValue = kurtosis(data);
disp(['Kurtosis = ', num2str(kurtosisValue)])

% Create a Q-Q plot
figure('Position', [15, 10, 1500, 1000]);
qqplot(data);
%title('Q-Q Plot for Log Returns');
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
set(gca, 'FontSize', 45);

%% Test of white noise, randomness, IID

% Compute the sample ACF
[acf, lags] = autocorr(data);

% Plot the ACF
figure('Position', [15, 10, 1500, 1000]);
stem(lags, acf, 'filled', 'MarkerSize', 5);
%title('Autocorrelation Function (ACF)');
xlabel('Lag');
ylabel('ACF');
set(gca, 'FontSize', 45);

% Perform the Turning Point Test
count = numel(find(diff(sign(data)) ~= 0));
disp(['Number of sign changes: ', num2str(count)]);
turning_points_ratio = count/length(data);
disp(['Turning point ratio = ', num2str(turning_points_ratio)])

% Perform the Rank test
% Initialize a counter
count = 0;

% Iterate through the data
for i = 1:(length(data) - 1)
    if data(i+1) > data(i)
        count = count + 1;
    end
end

rank_ratio = count/5784;
fprintf('Number of times data(i+1) > data(i): %d\n', count);
disp(rank_ratio);
%% Test of independent

% Perform the Durbin-Watson test
index_vector = 1:numel(AAPL.Date);
[p_dw, DW] = dwtest(data, index_vector);

% Display the results
disp(DW);
disp(p_dw);
if p_dw < 0.05
    disp('The returns are not independent.');
else
    disp('The returns are independent.');
end

% Perform the Ljung-Box Q-test
[h_lj, pValue_lj, stat, cValue] = lbqtest(data,Lags=(14));

% Display the results
disp(pValue_lj);
disp(h_lj);
if h
    disp('The data suggests significant autocorrelation.');
else
    disp('Insufficient evidence to autocorrelation.');
end