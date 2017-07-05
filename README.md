# S2SCycleGAN
Attempt at speech2speech using CycleGAN

# Audio Generation
Audio is converted using SEGAN method

# Data
Download data with `downloadData.sh`


## Port Audio and PyAudio

sudo apt-get install libasound-dev
wget http://portaudio.com/archives/pa_stable_v190600_20161030.tgz
tar -zxvf pa_stable_v190600_20161030.tgz
cd portaudio
./configure && make
sudo make install
cd ..
source activate tensorflow1
pip install pyaudio

