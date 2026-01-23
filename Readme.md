To run AlphaFace Demo, please conduct the following instruction.

For train.

1. install requirement (pip -r requirement.txt)
2. Run 'train_clip.sh'


For demo (Assuming you already installed the requirements)

1. Dwonload model file from `https://drive.google.com/file/d/18ZOQB3WmIFnMwi1GqBroFFEOuSNKWpZQ/view?usp=sharing' and put it in the project root directory.
2. Downlaod arcface model from 'https://drive.google.com/file/d/1qc4s6eRQPluma72WFibUnw74GPMAYRtY/view?usp=drive_link' and put it in the 'Models' directory. 
 - You may can change the configuration of model file path (config.model_path) in 'config/alphaface_eval_demo.py'
3. Run eval.sh

* We only provide data samples for training and evalaution demo (256 triplets of image, mask, and text)
