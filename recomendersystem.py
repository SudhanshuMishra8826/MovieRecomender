import numpy as np
#Importibng lightfm so we cqan import the movielens dataset
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
#fetch mivie data having a minnimum rating of four
data=fetch_movielens(min_rating=4.0)
#print(repr(data['train']))
#print(repr(data['test']))
#initialized lightfm class with loss parameter being warp : Weighted Approximate Rank pairwise
model=LightFM(loss='warp')
#train our model using fit method
model.fit(data['train'],epochs=30,num_threads=2)
#recomender function having model data nd list of userids as arguments and prointing the recomended movies
def Recomender(model,data,userids):
    #number of movies and users in our training data
    n_users,n_items=data['train'].shape
    #genertae recomendations for each user
    for user in userids:
        # list of movies they like
        known=data['item_labels'][data['train'].tocsr()[user].indices]
        #our predictions
        scores=model.predict(user,np.arange(n_items))
        #ranking them in non increasing order
        top=data['item_labels'][np.argsort(-scores)]
        #display results
        print("User %s"%user)
        print("------Known Choices:")
        for x in known[:3]:
            print("               %s"%x)
        print("------Recomended:")
        for x in top[:3]:
            print("               %s"%x)
#test call
Recomender(model,data,[2,25,200])

