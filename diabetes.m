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

g=(dataTrain.Glucose==0);
msa=(dataTrain.Outcome==1);
ks=mean(dataTrain.Glucose(~g & msa));
dataTrain.Glucose(g & msa)=ks;
ks=mean(dataTrain.Glucose(~g & ~msa));
dataTrain.Glucose(g & ~msa)=ks;
g=(dataTrain.BloodPressure==0);
ks=mean(dataTrain.BloodPressure(~g & msa));
dataTrain.BloodPressure(g & msa)=ks;
ks=mean(dataTrain.BloodPressure(~g & ~msa));
dataTrain.BloodPressure(g & ~msa)=ks;
g=(dataTrain.SkinThickness==0);
ks=mean(dataTrain.SkinThickness(~g & msa));
dataTrain.SkinThickness(g & msa)=ks;
ks=mean(dataTrain.SkinThickness(~g & ~msa));
dataTrain.SkinThickness(g & ~msa)=ks;
g=(dataTrain.Insulin==0);
ks=mean(dataTrain.Insulin(~g & msa));
dataTrain.Insulin(g & msa)=ks;
ks=mean(dataTrain.Insulin(~g & ~msa));
dataTrain.Insulin(g & ~msa)=ks;
g=(dataTrain.BMI==0);
ks=mean(dataTrain.BMI(~g & msa));
dataTrain.BMI(g & msa)=ks;
ks=mean(dataTrain.BMI(~g & ~msa));
dataTrain.BMI(g & ~msa)=ks;
g=(dataTrain.DiabetesPedigreeFunction==0);
ks=mean(dataTrain.DiabetesPedigreeFunction(~g & msa));
dataTrain.DiabetesPedigreeFunction(g & msa)=ks;
ks=mean(dataTrain.DiabetesPedigreeFunction(~g & ~msa));
dataTrain.DiabetesPedigreeFunction(g & ~msa)=ks;
dataTrain.Age(p)=mean(dataTrain.Age(~p));
%%
%Handle the 0 values for some features of test set
g=(dataTest.Glucose==0);
k=(dataTest.BloodPressure==0);
l=(dataTest.SkinThickness==0);
m=(dataTest.Insulin==0);
n=(dataTest.BMI==0);
o=(dataTest.DiabetesPedigreeFunction==0);
p=(dataTest.Age==0);
impractical=[sum(g) sum(k) sum(l) sum(m) sum(n) sum(o) sum(p)];
%%
%Replace those with average of the corresponding columns of test data
g=(dataTest.Glucose==0);
msa=(dataTest.Outcome==1);
ks=mean(dataTest.Glucose(~g & msa));
dataTest.Glucose(g & msa)=ks;
ks=mean(dataTest.Glucose(~g & ~msa));
dataTest.Glucose(g & ~msa)=ks;
g=(dataTest.BloodPressure==0);
ks=mean(dataTest.BloodPressure(~g & msa));
dataTest.BloodPressure(g & msa)=ks;
ks=mean(dataTest.BloodPressure(~g & ~msa));
dataTest.BloodPressure(g & ~msa)=ks;
g=(dataTest.SkinThickness==0);
ks=mean(dataTest.SkinThickness(~g & msa));
dataTest.SkinThickness(g & msa)=ks;
ks=mean(dataTest.SkinThickness(~g & ~msa));
dataTest.SkinThickness(g & ~msa)=ks;
g=(dataTest.Insulin==0);
ks=mean(dataTest.Insulin(~g & msa));
dataTest.Insulin(g & msa)=ks;
ks=mean(dataTest.Insulin(~g & ~msa));
dataTest.Insulin(g & ~msa)=ks;
g=(dataTest.BMI==0);
ks=mean(dataTest.BMI(~g & msa));
dataTest.BMI(g & msa)=ks;
ks=mean(dataTest.BMI(~g & ~msa));
dataTest.BMI(g & ~msa)=ks;
g=(dataTest.DiabetesPedigreeFunction==0);
ks=mean(dataTest.DiabetesPedigreeFunction(~g & msa));
dataTest.DiabetesPedigreeFunction(g & msa)=ks;
ks=mean(dataTest.DiabetesPedigreeFunction(~g & ~msa));
dataTest.DiabetesPedigreeFunction(g & ~msa)=ks;
dataTest.Age(p)=mean(dataTest.Age(~p));
%%

%Train model using training sets
classification_model=fitcsvm(dataTrain,'Outcome~Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction+Age');
%%
%Accuracy
gs=dataTest(:,1:8); %as 9th column is outcome%
mk=predict(classification_model,gs);
accurancy_check=((sum((mk==table2array(dataTest(:,9)))))/size(dataTest,1))*100;
disp(accurancy_check);

