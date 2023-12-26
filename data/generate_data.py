from render_model import ShapenetRender
import sys
import os
import subprocess

if __name__ == "__main__":
    inp_dir_path = sys.argv[1]
    out_path = sys.argv[2]
    path = os.path.join(os.getcwd(),inp_dir_path)
    model_path_list = [os.path.join(path,f) for f in os.listdir(inp_dir_path) 
                             if not os.path.isfile(os.path.join(path,f))]
    #model_path_list = [path]
    # print(model_path_list[0])
    # file = open('model_count.txt','r')
    # lines = file.readlines()
    # model_path_list = []
    # for i in range(int(len(lines)/2)):
    #     print(lines[2*i+1])
    #     if int(lines[2*i+1]) >=400:
    #         model_path_list.append(lines[2*i].strip()[:-1])

    num_images = sys.argv[3]


    for model_path in model_path_list:
        subprocess.run(['python3', 'generate_data_model.py',model_path,out_path,num_images])