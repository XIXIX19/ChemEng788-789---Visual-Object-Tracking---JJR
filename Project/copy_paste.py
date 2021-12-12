"""foldername = 'Woman'
savefolder = 'C:/Users/lir58/Desktop/Jierui/data'
open1 = savefolder + '/' + foldername + '_dataset.pkl'
open2 = savefolder + '/' + foldername + '_labelset.pkl'
with open(open1, 'rb') as file:
    template = pickle.load(file)
    target = pickle.load(file)
    target_0 = pickle.load(file)
with open(open2, 'rb') as file:
    label = pickle.load(file)
    label_0 = pickle.load(file)

template = np.vstack((template, template))
target = np.vstack((target, target_0))
label = np.vstack((label, label_0))"""