set -e

rm -rf keras-dcgan
rm -rf /tmp/logs
git clone https://github.com/ericguillaume/keras-dcgan.git
cd keras-dcgan
git checkout master
mkdir -p eric_generated
mkdir -p generated/model
python3 eric.py
