import os
import scipy.io
import numpy as np

names = ['1 NSR', '2 APB', '3 AFL', '4 AFIB', '5 SVTA', '6 WPW', '7 PVC', '8 Bigeminy', '9 Trigeminy', '10 VT', '11 IVR', '12 VFL', '13 Fusion', '14 LBBBB', '15 RBBBB', '16 SDHB', '17 PR']
for i in range(len(names)):
    l = os.listdir('MLII/' + names[i])
    arr = np.zeros((len(l),3600))
    nums = (i + 1) * np.ones((len(l),1))
    for j in range(len(l)):
        arr[j] = scipy.io.loadmat('MLII/' + names[i] + '/' + l[j])['val']
    arr = np.hstack((arr,nums))
    if i == 0:
        data = arr
    else:
        data = np.vstack((data,arr))
data = data.astype(int)
# np.savetxt('dataset/dataset.csv', data, delimiter=',')
# data.tofile('data/datatofile.csv',sep=',')
DF = pd.DataFrame(data) 
DF.to_csv('data/dataset.csv')
