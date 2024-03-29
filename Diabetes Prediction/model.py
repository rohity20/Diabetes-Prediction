
# In[1]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


dataset = pd.read_csv('diabetes.csv')


# # Step 3: Data Preprocessing

# In[13]:


dataset_X = dataset.iloc[:, [0, 1, 2, 4, 5, 7]].values

dataset_Y = dataset.iloc[:, 8].values


# In[14]:


dataset_X


# In[15]:


sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_X)


# In[16]:


dataset_scaled = pd.DataFrame(dataset_scaled)


# In[17]:


X = dataset_scaled
Y = dataset_Y


# In[18]:


X


# In[19]:


Y


# In[20]:


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42, stratify=dataset['Outcome'])


# # Step 4: Data Modelling

# In[25]:


svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train, Y_train)


# In[26]:


svc.score(X_test, Y_test)


# In[27]:


Y_pred = svc.predict(X_test)


pickle.dump(svc, open('model.sav', 'wb'))
model = pickle.load(open('model.sav', 'rb'))
print(model.predict(sc.transform(np.array([[1, 85, 66, 0, 26.6, 31]]))))
