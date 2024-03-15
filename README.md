## Get Started

1. Stage1: Fine-tune the model on single frame without using GCN.

    ```shell    
    # collective dataset
    python scripts/train_collective_stage1.py
    ```

2. Stage2: Fix weights of the feature extraction part of network, and train the network with GCN.

    ```shell
    # collective dataset
    update the stage1_model_path (path of the base model checkpoint) in config.py, and then run the script below:
    python scripts/train_collective_stage2.py
    ```

3. Test: Test the result using test video clips.
    ```shell
    # collective dataset
    update the stage2_model_path (path of the GCN model checkpoint) in config.py, and then run the script below:
    python scripts/test_collective.py
    ```
    
4. You can specify the running arguments in the python files under `scripts/` directory. The meanings of arguments can be found in `config.py`
   
   Based on our expirements, we suggest to use either NCC or SAD to calculate the pair-wise acotrs' appearance similarty, by setting the self.appearance_calc = "NCC" or "SAD":
      
   ![2](https://github.com/kuangzijian/Group-Activity-Recognition/blob/master/read_me_pictures/appearance_calc.png)

   To speed up the training/testing speed, we would also suggest to set the self.backbone='mobilenet':
   
   ![3](https://github.com/kuangzijian/Group-Activity-Recognition/blob/master/read_me_pictures/back_bone.png)
   
## Experiment results
   
1. Our expirements proved that using MobileNet as backbone in feature extraction will improve the training speed by 35%.

2. Our expirements proved that using NCC or SAD to calculate the pair-wise actors' appearance similarty and draw the actor relation graph can improve the group activity prediction accuracy.

   ![3](https://github.com/kuangzijian/Group-Activity-Recognition/blob/master/read_me_pictures/experiments.png)

## Acknowledgements


The majority of our project is based on J. Wu et al.'s work, our original contributions (are commented as "Original code by Zijian and Xinran" in each .py file) include changes:    
1. [backbone.py](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-Group-Activity-Recognition/blob/master/backbone.py)

2. [base_model.py](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-Group-Activity-Recognition/blob/master/base_model.py)

3. [collective.py](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-Group-Activity-Recognition/blob/master/collective.py)

4. [config.py](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-Group-Activity-Recognition/blob/master/config.py)

5. [gcn_model.py](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-Group-Activity-Recognition/blob/master/gcn_model.py)

6. [scripts/test_collective.py](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-Group-Activity-Recognition/blob/master/scripts/test_collective.py)

7. [scripts/train_collective_stage1.py](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-Group-Activity-Recognition/blob/master/scripts/train_collective_stage1.py)

8. [scripts/train_collective_stage2.py](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-Group-Activity-Recognition/blob/master/scripts/train_collective_stage2.py)

9. [test_net.py](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-Group-Activity-Recognition/blob/master/test_net.py)

10. [utils.py](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-Group-Activity-Recognition/blob/master/utils.py)

Paper Reference
```
@inproceedings{CVPR2019_ARG,
  title = {Learning Actor Relation Graphs for Group Activity Recognition},
  author = {Jianchao Wu and Limin Wang and Li Wang and Jie Guo and Gangshan Wu},
  booktitle = {CVPR},
  year = {2019}
}
```
Code Reference

https://github.com/wjchaoGit/Group-Activity-Recognition




