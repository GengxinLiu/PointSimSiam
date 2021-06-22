# Self-Supervised Learning on Point Clouds by Contrasting Global and Local Features(CAD & CG 2021)

### (1) Setup
This code has been tested with Python 3.5, Pytorch 1.5.
- Setup python environment
```
pip install -r helper_requirements.txt
```
 
 ### (2) Prepare ShapeNet and ModelNet40
- ShapeNet dataset can be found 
<a href="https://www.shapenet.org/">here</a>. 
Download the files named "ShapeNetCore.v2.zip". Then perform 
<a href="https://github.com/Salingo/virtual-3d-scanner">partial scanning</a>
on the mesh model from different views and save the results in `../ShapeNetScan` folder.

- ModelNet40 dataset can be found 
<a href="http://modelnet.cs.princeton.edu/">here</a>. 
Download the data and move it to `../ModelNet40` folder. 

**Note:** The data path can be changed in `data.py`.

 ### (3) Training
- train the self-supervised model on ShapeNet by 
```
sh sh_simsiam.sh
```

 ### (4) Linear SVM evaluation
- test the pretrain model using linear svm
```
python eval_linear.py
```
**Note:** You need to modify the model path.
