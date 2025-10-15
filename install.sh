conda create -n DPSS python=3.9
pip install torch torchvision torchaudio

pip install gdown mmcv==0.2.14 svgpathtools==1.6.1 munch==2.5.0 tensorboard==2.12.0 tensorboardx==2.5.1 

git clone https://github.com/facebookresearch/detectron2.git && cd detectron2 
python -m pip install .

cd ..

cd modules/pointops/src
sed -i 's:^\([^/].*\bTHC.*\)$://\1:' **/*.cpp # THC 只适合torch1.6 
cd ..
python setup.py install





