# Tensor-GAN

主文件为my_tgan

在my_util文件中有一个next_batch函数有用到

原数据按照label分别放到了label0_data,label1_data,label2_data文件夹下

在my_tgan中有data_path和output_path两个量分别表示读数据的路径和输出路径

运行时更改这两个参数即可

例如生成label为0的图片，则

```python
data_path = "./label1_data/"

output_path = "./eye_out_label1/"
```

core_h_dim为核张量第一维,core_w_dim为核张量第二维

gen_h_dim为prior的第一维，gen_w_dim为prior的第二维

核张量核prior的第三维都是3（我觉得通道数不变比较好）

