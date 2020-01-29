import os,shutil

label = open('save.csv','r').read().split('\n')[0:1830]
for line in label:
    data = line.split(',')
    data = [int(d) for d in data]
    filename = '{}_DiscCenter.png'.format(str(data[0]))
    if data[1] == 0:
        shutil.move('data_DiscCenter_enhance/'+filename, 'label0_data/'+filename)
    elif data[1] == 1:
        shutil.move('data_DiscCenter_enhance/' + filename, 'label1_data/' + filename)
    else:
        shutil.move('data_DiscCenter_enhance/' + filename, 'label2_data/' + filename)
