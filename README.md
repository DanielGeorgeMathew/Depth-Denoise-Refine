## Depth Denoising and Refinement using Multi-step CNN

This repository contains all code for training, validation, inference, data preparation/preprocessing for Denoising and RGB based Refinement of 
Noisy Depth Maps. The architecture is based on DDRNet : https://openaccess.thecvf.com/content_ECCV_2018/papers/Shi_Yan_DDRNet_Depth_Map_ECCV_2018_paper.pdf
and is implemented in Pytorch. 



## Setup

1. **Clone repository:**
```bash
https://github.com/DanielGeorgeMathew/Depth-Denoise-Refine.git

cd /path/to/this/directory
```

<br> 

----

2. **Set up the environment:** 

```bash
source update.sh
source install.sh
```

Cuda-11.x needs to be installed first

<br>

----

3. **Training/Validation Dataset preparation:** 

    1. Download and extract the desired RGB-D dataset: 
        For the refinement step, the albedo images have been generated using https://github.com/DreamtaleCore/USI3D and the Ground truth depth maps have been 
        obtained by projecting final Fused Mesh onto depth image plane for the respective frames. 
   
    2. Generate the train and test file paths to generate text files containing path to tran and validation sets, train.txt and val.txt, respectively.
        Also include in the same directory the instrinsic matrix numpy array stored as 'intrinsic.npy' of the device for which denoise/refine is performed. 
   3.  The train.txt/val.txt file needs to contain in each line the following:
        ***Path to RGB Image***, ***Path to original depth map***,    ***Path to Albedo Image***,   ***Path to Ground truth depth map***
   

<br>

----

4. **Training:**

Run the following command to start training:

```bash
python  train.py --data **Path to root folder containing train.txt, val.txt and intrinsic.npy** --batch_size 12 --run_path **Path to root folder
where tensorboard logs need to be stored** --model_save_path **Path to the root folder where model weights need to be stored** --num_epochs 100 --pateince 5 
--metadata_path **Path to the metadata.json file 
```

Run this command in a separate terminal to monitor training statistics and validation set results: 

```bash
tensorboard --logdir <path to resultsdir/logs> 
```
Open http://localhost:6006/ in browser to see training progress

<br>

----


5. **Inference:**


To use the trained model to denoise an entire folder of depth frames run:

```bash
python evaluate_depth_folder.py 
```

***Note: Modify root and model_weights_path variables in the above script to the Depth Fusion results root and path to the model checkpoint.pth 
respectively***

```bash
python evaluate_single_image.py 
```
***Note: Modify root, frame_id and weights_path variables in the above scripts respectively to the Kinect Fusion results root, depth frame number and path to the model checkpoint.pth 
respectively***


<br>

----