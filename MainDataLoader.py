import numpy as np


from DataLoader import *









if __name__ == "__main__":
    


    dl = DataLoader("./data/GMMData.mat", 1500, True)

    dl.reshuffle()

    for XtrainBatch, YtrainBatch in dl():
        print(XtrainBatch.shape)



    #for i in range(int(25000/1500)+1):
     #   XtrainBatch, YtrainBatch = dl.get_next_training_batch()

    

    dl.reshuffle()

