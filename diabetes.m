%load the data set%
dataset = csvread('diabetes.csv');
a = sum(isnull(dataset));

%FEATURE SCALING- x= x-avg or min/range
data.Pregnancies=(data.Pregnancies-min(data.Pregnancies))/(max(data.Pregnancies)-min(data.Pregnancies));
data.Glucose=(data.Glucose-min(data.Glucose))/(max(data.Glucose)-min(data.Glucose));
data.BloodPressure=(data.BloodPressure-min(data.BloodPressure))/(max(data.BloodPressure)-min(data.BloodPressure));
data.SkinThickness=(data.SkinThickness-min(data.SkinThickness))/(max(data.SkinThickness)-min(data.SkinThickness));
data.Insulin=(data.Insulin-min(data.Insulin))/(max(data.Insulin)-min(data.Insulin));
data.BMI=(data.BMI-min(data.BMI))/(max(data.BMI)-min(data.BMI));
data.DiabetesPedigreeFunction=(data.DiabetesPedigreeFunction-min(data.DiabetesPedigreeFunction))/(max(data.DiabetesPedigreeFunction)-min(data.DiabetesPedigreeFunction));
data.Age=(data.Age-min(data.Age))/(max(data.Age)-min(data.Age));


%training set partition 70/30%
%Split data into train test
cv = cvpartition(size(dataset,1),'HoldOut',0.3);
idx = cv.test;
% Separate to training and test data
dataTrain = dataset(~idx,:);
dataTest  = dataset(idx,:);
%%

%handling zeroes in data set%
g= (dataTrain.Glucose==0);
st = (dataTrain.SkinThickness==0);
bp = (dataTrain.BloodPressure==0);
i = (dataTrain.Insulin==0);
bmi = (dataTrain.BMI==0);
dpf = (dataTrain.DiabetesPedigreeFunction==0);
age = (dataTrain.Age==0);
impractical = [sum(g) sum(st) sum(bp) sum(i) sum(bmi) sum(dpf) sum(age)];

%replace 0 mean of benign and malignant%


%define sigmoid function%
function g = sigmoid(z)

g = zeros(size(z));



g= 1./(1+ exp(-z));

%cost function%



function [J, grad] = costFunction(theta, X, y)

m = length(y); %no of rows%

J = 0;
grad = zeros(size(theta));

J = (1 / m) * sum( -y'*log(sigmoid(X*theta)) - (1-y)'*log( 1 - sigmoid(X*theta)) );

grad = (1 / m) * sum( X .* repmat((sigmoid(X*theta) - y), 1, size(X,2)) );




