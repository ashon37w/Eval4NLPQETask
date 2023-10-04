from tool import get_zip
import numpy as np

if __name__ == "__main__": 
#     lens = open('zh_en_wizardLM_len_fu.scores', 'r').readlines()
#     scores = open('zh_en_wizard_demo.scores', 'r').readlines()
            
#     with open('zh_en_wizard_demo_mul.scores', 'w') as f:
#         for i in range(len(lens)):
#             f.write(str(int(scores[i]) * int(lens[i])) + "\n")
    
#     get_zip('zh_en_wizard_demo_mul.scores')
    
    scores1 = open('da_results/en_de_openorca_demo.scores', 'r').readlines()
    scores2 = open('da_results2/en_de_openorca_demo1.scores', 'r').readlines()
    scores3 = open('mqm_results/en_de_openorca_demo1.scores', 'r').readlines()
    scores4 = open('placeholder_results/en_de_openorca_demo1.scores', 'r').readlines()
    scores5 = open('min_max_results/en_de_openorca_demo1.scores', 'r').readlines()
    
    scores = []
    for i in range(len(scores1)):
        scores.append([int(scores1[i].strip()), int(scores2[i].strip()), int(scores3[i].strip()), int(scores4[i].strip()), int(scores5[i].strip())])
    
    mean_scores = []
    var_scores = []
    for score in scores:
        mean_scores.append(np.mean(score))
        var_scores.append(np.var(score))
        
    with open('en_de_openorca_mean.scores', 'w') as f:
        for score in mean_scores:
            f.write(str(score) + "\n")
    
    get_zip('en_de_openorca_mean.scores')
    
    with open('en_de_openorca_var.scores', 'w') as f:
        for score in var_scores:
            f.write(str(score) + "\n")
    
    get_zip('en_de_openorca_var.scores')
    
    
    scores1 = open('da_results/zh_en_openorca_demo.scores', 'r').readlines()
    scores2 = open('da_results2/zh_en_openorca_demo1.scores', 'r').readlines()
    scores3 = open('mqm_results/zh_en_openorca_demo1.scores', 'r').readlines()
    scores4 = open('placeholder_results/zh_en_openorca_demo1.scores', 'r').readlines()
    scores5 = open('min_max_results/zh_en_openorca_demo1.scores', 'r').readlines()
    
    scores = []
    for i in range(len(scores1)):
        scores.append([int(scores1[i].strip()), int(scores2[i].strip()), int(scores3[i].strip()), int(scores4[i].strip()), int(scores5[i].strip())])
    
    mean_scores = []
    var_scores = []
    for score in scores:
        mean_scores.append(np.mean(score))
        var_scores.append(np.var(score))
        
    with open('zh_en_openorca_mean.scores', 'w') as f:
        for score in mean_scores:
            f.write(str(score) + "\n")
    
    get_zip('zh_en_openorca_mean.scores')
    
    with open('zh_en_openorca_var.scores', 'w') as f:
        for score in var_scores:
            f.write(str(score) + "\n")
    
    get_zip('zh_en_openorca_var.scores')