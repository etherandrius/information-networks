python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="12,10,8,6,4,2,1"
python3.6 main_mi.py -e=1500 -s=15 -ts=0.05 -mie="bins" -bs=512 -ns="12,10,8,6,4,2,1"
python3.6 main_mi.py -e=1500 -s=15 -ts=0.85 -mie="bins" -bs=512 -ns="12,10,8,6,4,2,1"
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins,KDE,LNN_1,LNN_2,KSG" -bs=512 -ns="12,10,8,6,4,2,1"
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="12,10,8,6,4,2,1" -af=relu
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="12,10,8,6,4,2,1" -af=sigmoid
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="12,10,8,6,4,2,1" -af=linear
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=32 -ns="12,10,8,6,4,2,1"
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=64 -ns="12,10,8,6,4,2,1"
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=1024 -ns="12,10,8,6,4,2,1"
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="12,10,8,6,4,2,1"
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="12,10,10,8,2"
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="12,8,4"