
# coding: utf-8

# In[97]:

from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import csv
import matplotlib.pyplot as plt
import operator
get_ipython().magic(u'matplotlib inline')


# In[13]:

f = open('mailing_hw3.csv','rU')
reader = csv.reader(f, delimiter = ',')
dataset = []
for i in reader:
    dataset.append(i)
#target = [x[0] for x in reader]


# In[14]:

dataset = dataset[1:]


# In[15]:

target = [x[-1] for x in dataset]
train = [x[:-1] for x in dataset]


# In[16]:

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(train,target)


# In[88]:

f = open('mailing_hw3_natural.csv','rU')
reader = csv.reader(f, delimiter = ',')
test = []
for i in reader:
    test.append(i)
test1 = test[1:]
test = []
for i in test1:
    test.append(i[:-1])
test[0]    


# In[89]:

predicted_probs = [[index+1,x[1]] for index, x in enumerate(rf.predict_proba(test))]


# In[32]:

savetxt('Data/submission_natural.csv', predicted_probs, delimiter = ',', fmt = '%d,%f',header = 'CustomerId, PredictedProbability', comments = '')


# In[95]:

expected_value = []

for index, i in enumerate(rf.predict_proba(test)):
    expected_value.append([index+1,i[1]*float(test[index][0])])


# In[98]:

sorted_value = sorted(expected_value, key = operator.itemgetter(1), reverse = True)


# In[101]:

num_of_people = int(5000/0.68)
chosen_people = sorted_value[:num_of_people]
expected_donation = 0
for i in chosen_people:
    expected_donation+=i[1]


# In[102]:

print "Expected Profit from picking these people is",expected_donation


# In[ ]:



