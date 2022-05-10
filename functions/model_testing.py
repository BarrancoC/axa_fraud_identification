import pandas as pd
import pickle


def evaluate_prediction(Ytest, Ypredict):
    L = len(Ytest)
    TP = sum([Ytest[i]*Ypredict[i] for i in range(L)])
    FN = sum([Ytest[i]*(1-Ypredict[i]) for i in range(L)])
    TN = sum([(1-Ytest[i])*(1-Ypredict[i]) for i in range(L)])
    FP = sum([(1-Ytest[i])*Ypredict[i] for i in range(L)])
    accuracy = (TN+TP)/len(Ypredict)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    row = [TN, FP, TP, FN, accuracy, precision, recall]
    return row


def save_row(row, outputfile, print_row=False):
    row_str = ",".join([str(x) for x in row])
    if print_row:
        print(row_str)
    f = open(outputfile, "a")
    f.write(row_str+"\n")
    f.close()
    return


def sliding_window_test(mymodel, test_data, Y_col,
                        window=400, step=200,
                        outputfile='sliding_window_test.txt',
                        print_evaluations=False):
    L = len(test_data)
    N = L//step + 1
    header = "i0, i1, TN, FP, FN, TP, accuracy, precision, recall"
    if print_evaluations:
        print(header)
    f = open(outputfile, "w")
    f.write(header+"\n")
    f.close()
    for i in range(N):
        i0 = step*i
        i1 = i0+window
        if i1 <= L:
            dfi = test_data.iloc[i0: i1]
            actuals = list(dfi[Y_col])
            predictions = list(mymodel.predict(dfi.drop([Y_col],axis=1)))
            row = evaluate_prediction(actuals, predictions)
            save_row([i0, i1]+row, outputfile, print_evaluations)
    return


def rebalanced_sample_test(test_data, Y_col, size=1000, ratio=0.1):
    N1 = int(size*ratio)
    N2 = size - N1
    df1 = test_data[test_data[Y_col]==1]
    df2 = test_data[test_data[Y_col]==0]
    df1 = df1.sample(min(N1, len(df1)))
    df2 = df2.sample(min(N2, len(df2)))
    df = df1.append(df2).sort_index()
    return df


def sliding_window_rebalanced_sample_test(
    model_path, test_data_path, X_columns, Y_col,
    window=400, step=200,
    size=1000, ratio=0.1,
    outputfile='sliding_window_test.txt',
    print_evaluations=False
):
    mymodel = pickle.load(open(model_path, 'rb'))
    test_data = rebalanced_sample_test(
        pd.read_csv(test_data_path), Y_col, size, ratio) 
    columns = X_columns+[Y_col]
    sliding_window_test(
        mymodel, test_data[columns], Y_col, window, step, 
        outputfile, print_evaluations)
    return

    
    