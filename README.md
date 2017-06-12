# End-to-End Pooling (E2EP) Tensorflow Toolbox

E2EP is a Tensorflow toolbox that efficiently implements several learnable pooling method such as NetVLAD,
NetRVLAD, NetFV and Soft-DBoF as well as the Context Gating activation from: 

Miech et al., Learnable pooling method with Context Gating for video classification, arXiv.


## Usage example

Creating a NetVLAD block:

```python
import e2ep as ep

'''
Creating a NetVLAD layer with the following inputs:

feature_size: the dimensionality of the input features
max_samples: the maximum number of features per list
cluster_size: the number of cluster
output_dim: the dimensionality of the pooled features after 
dimension reduction
gating: If True, adds a Context Gating layer on top of the 
pooled representation
add_batch_norm: If True, adds batch normalization during training
is_training: If True, the graph is in training mode
'''
NetVLAD = ep.NetVLAD(feature_size=1024, max_samples=100, cluster_size=64, 
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

It is the same usage for NetRVLAD, NetFV and Soft-DBoF.

Antoine Miech
