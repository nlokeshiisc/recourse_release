from render_model import ShapenetRender
import sys
import os
import random

if __name__ == "__main__":
    # best_param_dict = {
    #                   "02691156" : [3,1,0],
    #                   "02828884" : [3,1,0],
    #                   "02924116" : [2,1,0],
    #                   "02933112" : [0,1,0],
    #                   "03001627" : [3,1,0],
    #                   "03211117" : [3,1,0],
    #                   "03624134" : [2,1,0],
    #                   "03636649" : [0,1,0],
    #                   "03691459" : [3,1,0],
    #                   "04090263" : [2,1,0],
    #                   }
    dist_dict = {0 : 0.5,
                1 : 1.5,
                2 : 4,
                }
    
    param_all = [[3,1,0],[2,1,0],[0,1,0],[1,0,1],[4,2,2],[5,2,3],[3,2,1],[2,0,2],[0,0,3]]
    #param_all = [[i,j,k] for i in range(6) for  j in range(3) for  k in range(4)]
    inp_dir_path = sys.argv[1]
    out_path = sys.argv[2]
    path = os.path.join(os.getcwd(),inp_dir_path)
    model_path_list = [os.path.join(path,f) for f in os.listdir(inp_dir_path)] 
                            #if not os.path.isfile(os.path.join(path,f))]
    #model_path_list = [path]
    print(model_path_list[0])

    num_images = int(sys.argv[3])
    model_path_list = random.sample(model_path_list,num_images)
    model_name = inp_dir_path.split("/")[-1]


    for model_path in model_path_list:
        model_instance_name = model_path.split("/")[-1]

        #best_param = best_param_dict[model_name]
        for param in param_all:
            out_model_path = out_path + "/" + model_name + "/" +model_instance_name +"{}_{}_{}".format(param[0],param[1],param[2])
            shapenetrender = ShapenetRender(model_path,out_model_path,param[0],dist_dict[param[1]],param[2])
            shapenetrender.render_acp(model_path+"/models/model_normalized.obj",out_model_path)
        
        # for i in range(3):
        #     out_model_path = out_path + "/" + model_name + "/" +"{}_{}_{}".format(best_param[0],dist_dict[i],best_param[2])
        #     shapenetrender = ShapenetRender(model_path,out_model_path,best_param[0],dist_dict[i],best_param[2])
        #     shapenetrender.render_random(int(0 if i==best_param[1] else 0))
        
        # for i in range(4):
        #     out_model_path = out_path + "/" + model_name + "/" +"{}_{}_{}".format(best_param[0],dist_dict[best_param[1]],i)
        #     shapenetrender = ShapenetRender(model_path,out_model_path,best_param[0],dist_dict[best_param[1]],i)
        #     shapenetrender.render_random(int(0 if i==best_param[2] else 0))
        # #for param in param_all:
        #     out_model_path = out_path + "/" + model_name + "/" +"{}_{}_{}".format(param[0],dist_dict[param[1]],param[2])
        #     shapenetrender = ShapenetRender(model_path,out_model_path,param[0],dist_dict[param[1]],param[2])
        #     shapenetrender.render_random(int(num_images*0.2 if param==best_param else num_images*0.1))
              
    
