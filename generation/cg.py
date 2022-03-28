import torch
model=torch.load('checkpoint/color_all_ae_64/IM_AE.model32-499.pth')
dic=torch.load('checkpoint/color_all_ae_64/IM_AE.model32-499.pth')

for k in model.keys():
  if 'linear_1.weight' in k:
    data=torch.zeros((1024,259+32))
    data[:,:259]=model[k]
    dic[k]=data
    #dic[k.replace('linear_1.weight','linear_1_cct.weight')]=data
    #if 'linear_1.bias' in k:
    #dic[k.replace('linear_1.bias','linear_1_cct.bias')]=model[k]
  else:
    dic[k]=model[k]

torch.save(dic,'checkpoint/color_all_ae_64/IM_AE.model32-499-cg.pth')
