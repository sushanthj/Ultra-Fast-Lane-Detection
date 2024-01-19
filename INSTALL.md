
# Install
1. Clone the project

    ```Shell
    git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection
    cd Ultra-Fast-Lane-Detection
    ```

2. Install dependencies

    - Inside a new venv environment
      ```bash
      pip3 install torch torchvision -f https://download.pytorch.org/whl/cu101/torch_stable.html
      pip3 install -r requirements.txt
      ```
    - Optionally install a few more requirements if there are some mising packages (recommended)
      ```bash
      pip3 install -r extended_requirements.txt
      ```

3. Data preparation for Custom Dataset

    - To understand how we can build the custom dataset to match CULane we first need to
    understand how CULane's format (Link to CULane : https://xingangpan.github.io/projects/CULane.html)
    - CULane has RGB images and binary mask images for training (see binary mask example below)
      ![](mask.png)
    - The dataloader for this model requires only the filepaths of the above images. To get
      these filepaths, the network looks for a train_gt.txt (train set) and test.txt (validation set)
    - This train_gt.txt and test.txt must located in a folder called **'list'** and the filepath
      of this 'list' folder must be added to the configs/young_soybean.py file as shown below
      ```python
        # The folder project_train_data_1 contains the 'list' folder as well as the training
        # validation, and testing images
        dataset='CULane'
        data_root = '/home/mrsd_teamh/sush/11-785/project_train_data_1'


        # TRAIN (Do not change, also SGD works better for this model)
        epoch = 50
        batch_size = 32
        optimizer = 'SGD'  #['SGD','Adam']
        learning_rate = 0.1
        weight_decay = 1e-4
        momentum = 0.9
      ```
    - For inference of images, we need to also specify image paths. However, these paths
      must be present in a folder called 'test_split' inside the 'list' folder
    - test_split folder will internally have test0_normal.txt, test1_normal.txt ..etc. which
      will contain the filepaths to inference images
    - Another folder called test_output will need to be present in the 'data_root' as per the tree shown below

    Here's a simple illustration of the desired structure of the 'list' folder

    ```
    project_train_data_1/
    ├── list
    │   ├── test.txt
    │   ├── test_split
    │   │   └── test0_normal.txt
    │   └── train_gt.txt
    |
    ├── training_images
    │   ├── images-clean
    │   │   ├── frame_0000.png
    │   │   └── frame_1140.png
    |   |
    │   └── masks-clean
    │       ├── frame_0000_mask.png
    │       └── frame_1140_mask.png
    |
    └── test_output (images after inference and drawing lanes will be saved here)
        ├── 0.png
        └── 9.png
    ```

4. Using Pre-Trained Model for Finetuning

    - The config file can be modified to use a model already pre-trained on the CULane
      dataset. Please download the necessary pre-trained model from : https://drive.google.com/file/d/1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq/view?usp=sharing
    - After downloading the model, update the 'Fine Tune' path in config as shown below
    - I used the 18 layer backbone as it is the lightest. I also had low training data
      and therefore could not finetune a larger model

    NOTE: We also need to create a folder to log all training (training will fail if not)

    ```python
    # NETWORK
    use_aux = True
    griding_num = 200
    backbone = '18'

    # LOSS
    sim_loss_w = 0.0
    shp_loss_w = 0.0

    # EXP
    note = ''

    # This folder must be created by user
    log_path = "/home/mrsd_teamh/sush/11-785/ufld_logging"

    # FINETUNE or RESUME MODEL PATH
    # finetune = None
    finetune = "/home/mrsd_teamh/sush/11-785/Ultra-Fast-Lane-Detection/checkpoints/culane_18.pth"
    resume = None

    # TEST
    test_model = "/home/mrsd_teamh/sush/11-785/Ultra-Fast-Lane-Detection/checkpoints/checkpoint.pth"
    test_work_dir = None

    num_lanes = 4
    ```