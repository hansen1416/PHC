sudo -v ; curl https://rclone.org/install.sh | sudo bash

rclone config

mkdir /mnt/gdrive_humos_output

git clone https://github.com/hansen1416/PHC.git

-------------------

sudo apt update

curl -O https://repo.anaconda.com/archive/Anaconda3-2025.12-2-Linux-x86_64.sh
bash ~/Anaconda3-2025.12-2-Linux-x86_64.sh


sudo -v ; curl https://rclone.org/install.sh | sudo bash
rclone config

mkdir /mnt/gdrive_humos_output

git clone https://github.com/hansen1416/PHC.git

conda create -n phc python=3.8 -y

conda activate phc
conda install -y pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 cpuonly -c pytorch
pip install numpy scipy tqdm

tmux new -s rclone

rclone mount gdrive:humos_output /mnt/gdrive_humos_output --read-only --vfs-cache-mode off --buffer-size 40M -vv

tmux new -s humos

python scripts/humos2phc_data_parallel.py --workers 32 2>&1 | tee humos2phc.log


# Then detach from tmux by pressing:

`Ctrl+b, then d`

# Later, reconnect with:

`tmux attach -t humos`

# unmount

findmnt /mnt/gdrive_humos_output
mount | grep gdrive_humos_output
ps -ef | grep 'rclone mount' | grep -v grep

fusermount -u /mnt/gdrive_humos_output



tmux kill-session -t humos
tmux kill-session -t rclone
