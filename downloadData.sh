mkdir data
cd data
mkdir male_us
mkdir female_us
cd female_us
wget -r -nH -nd --no-parent http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_slt_arctic/wav/
cd ..
cd male_us
wget -r -nH -nd --no-parent http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_bdl_arctic/wav/
cd ..
