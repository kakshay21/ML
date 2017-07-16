from keras.models import Sequential
from keras.layers import Dense
import numpy as np
seed=34
np.random.seed(seed)
dataset=np.genfromtxt("ionosphere.csv",delimiter=",")
X=dataset[:,0:-1]
Y=dataset[:,-1]
model=Sequential()
model.add(Dense(12,input_dim=34,init='uniform',activation='relu'))
model.add(Dense(8,init='uniform',activation='relu'))
model.add(Dense(1,init='uniform',activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,Y,nb_epoch=150,batch_size=10)
scores=model.evaluate(X,Y)
print("%s:%.2f%%"%(model.metrics_names[1],scores[1]*100))
