# S2SCycleGAN
Attempt at speech2speech using CycleGAN

# Audio Generation
Audio is converted using SEGAN method.  See nbs folder for project.

# Tacotron
Model originated from [here](https://github.com/Kyubyong/tacotron) and was updated to take audio as input.  The updated model will be used as the generator.  Work still needs to be done for the discriminator.

# Data
Download data with male and female data with `downloadData.sh`


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

