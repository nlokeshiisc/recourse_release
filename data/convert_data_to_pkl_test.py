from copyreg import pickle
from this import d
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import sys
import os
import numpy as np
import pickle
import random

def gen_tuple(model_img,model_name,best_param,instance):
    x = Image.open(model_img)
    x = np.array(x,dtype='short')
    x = np.transpose(x,(2,0,1)) 
    #x_tensor = TF.to_tensor(x)
    #best_param = best_param_dict[model_name]
    param = model_img[-9:-4].split("_")
    param = [int(i) for i in param]
    y = y_dict[model_name]
    z_path = model_img
    z_path = z_path[:-9]+"{}_{}_{}.png".format(best_param[0],best_param[1],best_param[2])
    z = Image.open(z_path)
    z = np.array(z,dtype='short')
    z = np.transpose(z,(2,0,1)) 
    
    #z_tensor = TF.to_tensor(z)
    beta = param
    siblings = []
    ideal_beta = 1 if best_param==param else 0
    ins = instance
    li = [x,z,beta,y,ins,siblings,ideal_beta]

    return li
    
def non_ideal_list(best_param,param_list):
    ni_list = []
    for param in param_list:
        if param != best_param:
            ni_list.append(param)
    return ni_list
    

if __name__ == "__main__":
    best_param_dict = {
                      "02691156" : [3,1,0],
                      "02828884" : [3,1,0],
                      "02924116" : [2,1,0],
                      "02933112" : [0,1,0],
                      "03001627" : [3,1,0],
                      "03211117" : [3,1,0],
                      "03624134" : [2,1,0],
                      "03636649" : [0,1,0],
                      "03691459" : [3,1,0],
                      "04090263" : [2,1,0],
                      }
    #param_list = [[3,1,0],[2,1,0],[0,1,0],[1,0,1],[4,2,2],[5,2,3],[3,2,1],[2,0,2],[0,0,3]]
    param_list = [[i,j,k] for i in range(6) for  j in range(3) for  k in range(4)]

    y_dict = {

                      "02691156" : 0,
                      "02828884" : 1,
                      "02924116" : 2,
                      "02933112" : 3,
                      "03001627" : 4,
                      "03211117" : 5,
                      "03624134" : 6,
                      "03636649" : 7,
                      "03691459" : 8,
                      "04090263" : 9,
                     
            }
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    
    data = []

    inp_dir_path = sys.argv[1]
    path = os.path.join(os.getcwd(),inp_dir_path)
    model_path_list = [os.path.join(path,f) for f in os.listdir(inp_dir_path) 
                            if not os.path.isfile(os.path.join(path,f))]


    instance = -1
    model_path_list = sorted(model_path_list)
    for model_path in model_path_list:
        model_img_list = [os.path.join(model_path,f) for f in os.listdir(model_path) 
                            if  os.path.isfile(os.path.join(model_path,f))]
        model_img_list = sorted(model_img_list)
        model_name = model_path.split("/")[-1]
        #best_param = best_param_dict[model_name]
        model_img_split_list = [list(i) for i in np.array_split(model_img_list,int(sys.argv[2]))]
        #model_non_ideal_list = non_ideal_list(best_param,param_list)
        for model_img_type_list in model_img_split_list:
            instance = instance+1
            model_img_type_path = model_img_type_list[0][:-9]    
            for param in param_list:
                model_img = model_img_type_path+"{}_{}_{}".format(param[0],param[1],param[2])+".png"
                data.append(gen_tuple(model_img,model_name,[3,1,0],instance))
                ##    

                

                    

    print(data[0][2:])
    #data_t = torch.tensor(data)
    with open(sys.argv[3],"wb") as file:
        pickle.dump(data,file)
    #torch.save(data,"data.pkl")
            










    
