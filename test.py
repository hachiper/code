from autogluon.tabular import TabularDataset,TabularPredictor
import pandas as pd
train_data=TabularDataset("train.csv")
id,label="id","target"
predictor=TabularPredictor(label=label).fit(train_data.drop(columns=[id]))
test_date=TabularDataset('test.csv')
preds=predictor.predict(test_date.drop(columns=[id]))
proba=predictor.predict_proba(test_date.drop(columns=[id]))
proba.to_csv("submissio.csv",index=False)
#submission=pd.DataFrame({id:test_date[id], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3],'Class_5':proba[:,4],'Class_6':proba[:,5],'Class_7':proba[:,6],'Class_8':proba[:,7],'Class_9':proba[:,8]})
#submission.to_csv('submission.csv',index=False)
