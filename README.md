# Learnable mOdUle for Pooling fEatures (LOUPE) Tensorflow Toolbox

LOUPE is a Tensorflow toolbox that efficiently implements several learnable pooling method such as NetVLAD[1],
NetRVLAD[2], NetFV[2] and Soft-DBoW[2] as well as the Context Gating activation from: 

Antoine Miech and Ivan Laptev and Josef Sivic, Learnable pooling with Context Gating for video classification, arXiv:1706.06905, 2017.

It was initially used by the winning approach of the Youtube 8M Kaggle Large-Scale Video understading challenge:
 https://www.kaggle.com/c/youtube8m. We however think these are some general pooling approaches that can be used
in various applications other than video representation. That is why we decided to create this small Tensorflow toolbox.


## Usage example

Creating a NetVLAD block:

```python
import loupe as lp

'''
Creating a NetVLAD layer with the following inputs:

feature_size: the dimensionality of the input features
max_samples: the maximum number of features per list
cluster_size: the number of clusters
output_dim: the dimensionality of the pooled features after 
dimension reduction
gating: If True, adds a Context Gating layer on top of the 
pooled representation
add_batch_norm: If True, adds batch normalization during training
is_training: If True, the graph is in training mode
'''
NetVLAD = lp.NetVLAD(feature_size=1024, max_samples=100, cluster_size=64, 
                     output_dim=1024, gating=True, add_batch_norm=True,
                     is_training=True)



'''
Forward pass of the pooling architecture with
tensor_input: A tensor of shape:
'batch_size'x'max_samples'x'feature_size'
tensor_output: The pooled representation of shape:
'batch_size'x'output_dim'
'''
tensor_output = NetVLAD.forward(tensor_input)
```
It is the same usage for NetRVLAD, NetFV and Soft-DBoW.

NOTE: The toolbox can only pool lists of features of the same length.
It was specifically optimized to efficiently do so.
One way to handle multiple lists of features of variable length
is to create, via a data augmentation technique, a tensor of shape: 'batch_size'x'max_samples'x'feature_size'.
Where 'max_samples' would be the maximum number of feature per list.
Then for each list, you would fill the tensor with 0 values.

## References

[1] Arandjelovic, Relja and Gronat, Petre and Torii, Akihiko and Pajdla, Tomas and Sivic, Josef, NetVLAD: CNN architecture for weakly supervised place recognition, CVPR 2016

If you use this toolbox, please cite the following paper:

[2] Antoine Miech and Ivan Laptev and Josef Sivic, Learnable pooling with Context Gating for video classification, arXiv:1706.06905:
```
@article{miech17loupe,
  title={Learnable pooling with Context Gating for video classification},
  author={Miech, Antoine and Laptev, Ivan and Sivic, Josef},
  journal={arXiv:1706.06905},
  year={2017},
}
```



Antoine Miech
