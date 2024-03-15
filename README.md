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
    
