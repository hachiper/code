import numpy as np
import pandas as pd

sub1=pd.read_csv("1.74410 version 44.csv")
sub2=pd.read_csv("1.74412 version 44.csv")
sub3=pd.read_csv("1.74412 version 47.csv")
sub4=pd.read_csv("1.74413 version 45.csv")
subn=pd.read_csv("version 18 further 1.74404.csv")
subk=pd.read_csv("comparative.csv")
subt=pd.read_csv("ensembling.csv")
subgg=pd.read_csv("sample_submission.csv")
columns=sub1.columns[1:]
blend=sub1.copy()
blend[columns]=0.80*0.6*subk[columns]+0.80*0.4*subt[columns]+0.10*0.5*sub3[columns]+0.10*0.5*sub4[columns]+0.10*subgg[columns]
blend.to_csv("a.csv",index=False)
blen=subn.copy()
blen[columns]=1.01*subn[columns]-0.01*blend[columns]
blen.to_csv("a_+.csv",index=False)
blend[columns]=0.80*0.6*subk[columns]+0.80*0.4*subt[columns]+0.10*0.5*sub1[columns]+0.10*0.5*sub2[columns]+0.10*subgg[columns]
blend.to_csv("ak_6.csv",index=False)

blend[columns]=0.85*0.6*subk[columns]+0.85*0.4*subt[columns]+0.15*0.5*sub3[columns]+0.15*0.5*sub4[columns]
blend.to_csv("ak_9.csv",index=False)


