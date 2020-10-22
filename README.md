# Magnetic_tile_defect_prediction

## REST API SETUP
### Download Anaconda 5.3.0 or above
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
### Install Anaconda
bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh

### Create conda environment
conda create -n mag python=3.6 anaconda

### Activate the environment
conda activate mag

### Clone the repo
git clone https://github.com/Dharani1008/Magnetic_tile_defect_prediction/

### Install the dependencies
pip install -r requirements.txt

### run the api server
python flask_sever.py

### Test the api server

cURL command example:
`curl -F "image=@0_L_CC.png" -F "model=2"  http://{ipaddress}:5012/api/v0`



