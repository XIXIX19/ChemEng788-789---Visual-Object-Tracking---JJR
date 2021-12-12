import numpy as np

"""a = [[1,3,5],[2,3,4],[3,3,9]]
a = np.array(a)
b = np.expand_dims(a,axis=0)
c = np.amax(a,axis=(0,1))
"""
a = 1
b = 2
c = 3
d = np.array([a,b,c])
new_id = np.argmax(d)
print(new_id)

a = [1,2,3]
b = [4,5,6]
c = [7,8,9]
update_scoremap = [ [1,2,3],
                    [4,5,6],
                    [7,8,9]]
update_scoremap = np.array(update_scoremap )
p = np.asarray(np.unravel_index(np.argmax(update_scoremap), np.shape(update_scoremap)))
print(p)