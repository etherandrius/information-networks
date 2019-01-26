python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="10,8,6,4"
echo "Subject: machine2 \n\n 1/14" | ssmtp ag939@cam.ac.uk
python3.6 main_mi.py -e=1500 -s=15 -ts=0.05 -mie="bins" -bs=512 -ns="10,8,6,4"
echo "Subject: machine2 \n\n 2/14" | ssmtp ag939@cam.ac.uk
python3.6 main_mi.py -e=1500 -s=15 -ts=0.85 -mie="bins" -bs=512 -ns="10,8,6,4"
echo "Subject: machine2 \n\n 3/14" | ssmtp ag939@cam.ac.uk
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins,KDE,LNN_1,LNN_2,KSG" -bs=512 -ns="10,8,6,4"
echo "Subject: machine2 \n\n 4/14" | ssmtp ag939@cam.ac.uk
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="10,8,6,4" -af=relu
echo "Subject: machine2 \n\n 5/14" | ssmtp ag939@cam.ac.uk
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="10,8,6,4" -af=sigmoid
echo "Subject: machine2 \n\n 6/14" | ssmtp ag939@cam.ac.uk
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="10,8,6,4" -af=linear
echo "Subject: machine2 \n\n 7/14" | ssmtp ag939@cam.ac.uk
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=32 -ns="10,8,6,4"
echo "Subject: machine2 \n\n 8/14" | ssmtp ag939@cam.ac.uk
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=64 -ns="10,8,6,4"
echo "Subject: machine2 \n\n 9/14" | ssmtp ag939@cam.ac.uk
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=1024 -ns="10,8,6,4"
echo "Subject: machine2 \n\n 10/14" | ssmtp ag939@cam.ac.uk
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="10,8,6,4"
echo "Subject: machine2 \n\n 11/14" | ssmtp ag939@cam.ac.uk
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="10,10"
echo "Subject: machine2 \n\n 12/14" | ssmtp ag939@cam.ac.uk
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="Dr,10,Dr,8,Dr,6,Dr,4,Dr"
echo "Subject: machine2 \n\n 13/14" | ssmtp ag939@cam.ac.uk
python3.6 main_mi.py -e=1500 -s=15 -ts=0.45 -mie="bins" -bs=512 -ns="8,4"
echo "Subject: machine2 \n\n 14/14 Batch job is done!" | ssmtp ag939@cam.ac.uk
