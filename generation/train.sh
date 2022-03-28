python -u main.py --ae --train --epoch 500 --sample_vox_size 16
python -u main.py --ae --train --epoch 500 --sample_vox_size 32 --initialize checkpoint/color_all_ae_64/IM_AE.model16-499.pth
python cg.py
python -u main.py --o2o --train --epoch 200 --sample_vox_size 32 --initialize checkpoint/color_all_ae_64/IM_AE.model32-499-cg.pth
python -u main.py --res64 --train --epoch 200 --sample_vox_size 64 --initialize checkpoint/color_all_ae_64/o2o.model32-199.pth
python -u main.py --div --train --epoch 150 --sample_vox_size 64 --initialize checkpoint/color_all_ae_64/res64.model64-199.pth