df = readtable('Data_For_Use.csv');
X = [table2array(df(:,"x__DebyeAng_")),table2array(df(:,"x__DebyeAng3_"))];
y = [table2array(df(:,"Tg_K_"))];
n = 60;
l = 50;
indexes = [1:60];
train_indexes = randperm(n,l);
X_train = X(train_indexes,:);
y_train = y(train_indexes,:);
indexes(train_indexes) = [];
X_test = X(indexes,:);
y_test = y(indexes,:);
mdl = fitrgp(X_train,y_train,'OptimizeHyperparameters','all');
y_predicted_train = mdl.predict(X_train);
y_predicted_test = mdl.predict(X_test);
train_corr = corrcoef(y_train,y_predicted_train);
test_corr = corrcoef(y_test,y_predicted_test);
train_error = y_predicted_train - y_train;
test_error = y_predicted_test - y_test;










