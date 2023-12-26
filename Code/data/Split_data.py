import sys
import os
import subprocess
import random
if __name__ == "__main__":
    #inp_dir_path = sys.argv[1]
    out_path = sys.argv[1]
    #path = os.path.join(os.getcwd(),inp_dir_path)
    # model_path_list = [os.path.join(path,f) for f in os.listdir(inp_dir_path) 
    #                         if not os.path.isfile(os.path.join(path,f))]
    # #model_path_list = [path]
    #print(model_path_list[0])
    file = open('model_count.txt','r')
    lines = file.readlines()
    model_path_list = []
    for i in range(int(len(lines)/2)):
        #print(lines[2*i+1])
        if int(lines[2*i+1]) >=400:
            model_path_list.append(lines[2*i].strip()[:-1])


    train = int(sys.argv[2])
    val = int(sys.argv[3])
    test = int(sys.argv[4])




    for model_path in model_path_list:
        model_name = model_path.split('/')[-1]
        model_list = [os.path.join(model_path,f) for f in os.listdir(model_path) 
                            if not os.path.isfile(os.path.join(model_path,f))]
        model_list_random = random.sample(model_list,train+val+test)
        os.system("mkdir -p "+out_path+"/train/"+model_name)
        os.system("mkdir -p "+out_path+"/val/"+model_name)
        os.system("mkdir -p "+out_path+"/test/"+model_name)
        
        for model in model_list_random[:train]:
            #print(model_path.split("/")[-1])
            #print(model.split("/")[-1])
            os.system("mkdir -p "+out_path+"/train/"+model_name)

            os.system("cp -r "+model+" "+out_path+"/train/"+model_name)
        for model in model_list_random[train:train+val]:
            #print(model_path.split("/")[-1])
            #print(model.split("/")[-1])
            os.system("mkdir -p "+out_path+"/val/"+model_name)

            os.system("cp -r "+model+" "+out_path+"/val/"+model_name)
        for model in model_list_random[train+val:]:
            #print(model_path.split("/")[-1])
            os.system("mkdir -p "+out_path+"/test/"+model_name)
            #print(model.split("/")[-1])
            os.system("cp -r "+model+" "+out_path+"/test/"+model_name)
            



        