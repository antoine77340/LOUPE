# End-to-End Pooling (E2EP) Tensorflow Toolbox

E2EP is a Tensorflow toolbox that efficiently implements several learnable pooling method such as NetVLAD,
NetRVLAD, NetFV and Soft-DBoF as well as the Context Gating activation from: 

Miech et al., Learnable pooling method with Context Gating for video classification, arXiv.


## Usage example

Creating a NetVLAD block:

```python
import e2ep as ep

NetVLAD = ep.NetVLAD(feature_size=1024, max_samples=100, cluster_size=64, 
                     output_dim=1024, gating=True, add_batch_norm=True,
                     is_training=True)


output = NetVLAD.forward(input)
```

Antoine Miech
