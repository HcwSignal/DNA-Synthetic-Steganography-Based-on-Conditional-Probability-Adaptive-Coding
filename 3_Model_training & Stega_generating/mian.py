import torch
from torch import nn
import torch.optim as optim
import scipy.stats
import numpy as np
from logger import Logger
import utils
import lm
import os
import inspect
import Model_training
import remove_model
import stega

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def claer_dirs(dirs_modes):
    del_list = os.listdir(dirs_modes)
    for f in del_list:
        file_path = os.path.join(dirs_modes, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    File = []

    mode = '572' #This is the code name for the real sequence sample

    file_name = r'ena_438_noncoding'#This is also the code name for the real sequence sample

    Path_model_save = r'D:\Desktop\PaperCode_2023\ExperimentData' #root dirs

    Range = [1, 2, 3]  # repeated experiment

    index = ['2','3','4', '5','6','8','10','20','33','40','50'] # SL value

    Clear_File = False #Clear all models in the model save folder

    for ind in index:
        code = 'fxy' + ind
        if int(ind) % 3 == 0:
            SeqLength = int(198 / int(ind)) #the number of base unit in single short sequence
            file_ = Path_model_save + r'\{}\base_extaction\{}_198_{}.txt'.format(mode, file_name, ind) #original DNA sequence dataset
        else:
            SeqLength = int(200 / int(ind))
            file_ = Path_model_save + r'\{}\base_extaction\{}_200_{}.txt'.format(mode, file_name, ind)

        for i in Range:
            os.makedirs( Path_model_save + r'\{}\read_{}\M_{}'.format(mode, ind, str(i)), exist_ok=True) #create the model saving path

            dirs_modes = Path_model_save + r'\{}\read_{}\M_{}'.format(mode, ind,str(i))

            if Clear_File == True:
                claer_dirs(dirs_modes) #Delete all saved models in the folder
            else:
                continue

            Model_training.main(file_, code, SeqLength, ind, Path_model_save, mode, modelnum=i, lr=0.0001, running_mode='t')
            # The redundant model is deleted and the model when the loss function converges is retained
            remove_model.qingchu_Model(Path_model_save + r"\{}\read_{}\M_{}".format(mode, ind, str(i)))


        # The secret information embedding and pesudo-sequence generation begin
        stega.main_stega(mode=mode, Ranges=Range, index=[ind], file_=file_, Path_save=Path_model_save, test=False, coding='adg')






