set -e

rm -rf keras-dcgan
rm -rf /tmp/logs
rm -rf /Users/eric/dev/data/anime-faces-processed_5k

mkdir -p /home/eric/dev/data
cd /home/eric/dev/data
tar -xzvf /tmp/anime-faces-processed_5k.tar.gz

cd /home/eric/
git clone https://github.com/ericguillaume/keras-dcgan.git
cd keras-dcgan
git checkout anime
mkdir -p eric_generated
mkdir -p generated/model
python3 eric.py
