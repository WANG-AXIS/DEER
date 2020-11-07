# Deep Efficient End-to-end Reconstruction (DEER) Network for Few-view Breast CT Image Reconstruction
To start the training process, please prepare your training/validation/testing data in the following format

* ``img_data`` (Few view images, FDK, FBP, etc): N * 1 * img_width * img_height
* ``projection_data``: N * 1 * num_views * num_detector
* ``label``: N * 1 * img_width * img_height

Here, N represents the number of input data.
All datasets are stored in hdh5 files.

## Citation
If you found this code or our work useful, please cite us.

Paper DOI: https://doi.org/10.1109/ACCESS.2020.3033795
```
@ARTICLE{9239986,  author={H. {Xie} and H. {Shan} and W. {Cong} and C. {Liu} and X. {Zhang} and S. {Liu} and R. {Ning} and G. {Wang}}, 
journal={IEEE Access}, 
title={Deep Efficient End-to-End Reconstruction (DEER) Network for Few-View Breast CT Image Reconstruction}, 
year={2020},
volume={8},
number={}, 
pages={196633-196646},
doi={10.1109/ACCESS.2020.3033795}}
```
## Contact
huidong.xie at ieee dot org

Any discussions, suggestions and questions are welcome!
