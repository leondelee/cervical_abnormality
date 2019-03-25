#!/usr/bin/env python
# coding: utf-8

# In[2]:


from data_loader import *
x,y = load_image_svm(input_size = 540)
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2)
clf = SVC(kernel='linear',C=1,gamma = "auto")
clf.fit(train_x,train_y)
pred_y = clf.predict(test_x)
print(classification_report(test_y,pred_y))


# In[ ]:




