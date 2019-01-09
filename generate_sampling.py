import sys
niftynet_path = '../../'
sys.path.append(niftynet_path)


from niftynet.io.image_reader import ImageReader
from niftynet.engine.sampler_uniform_v2 import UniformSampler
from niftynet.layer.pad import PadLayer
from niftynet.layer.rand_elastic_deform import RandomElasticDeformationLayer
from niftynet.layer.mean_variance_normalisation import MeanVarNormalisationLayer
from niftynet.layer.rand_flip import RandomFlipLayer

def create_image_reader(num_controlpoints, std_deformation_sigma):
    # creating an image reader.
    data_param = \
        {'cell': {'path_to_search': '../../data/u-net/PhC-C2DH-U373/niftynet_data', # PhC-C2DH-U373, DIC-C2DH-HeLa
                'filename_contains': 'img_',
                'loader': 'skimage'},
         'label': {'path_to_search': '../../data/u-net/PhC-C2DH-U373/niftynet_data', # PhC-C2DH-U373, DIC-C2DH-HeLa
                'filename_contains': 'bin_seg_',
                'loader': 'skimage',
                'interp_order' : 0}
        }
    reader = ImageReader().initialise(data_param)

    reader.add_preprocessing_layers(MeanVarNormalisationLayer(image_name = 'cell'))

    reader.add_preprocessing_layers(PadLayer(
                     image_name=['cell', 'label'],
                     border=(92,92,0),
                     mode='symmetric')) 

    reader.add_preprocessing_layers(RandomElasticDeformationLayer(
                     num_controlpoints=num_controlpoints,
                     std_deformation_sigma=std_deformation_sigma,
                     proportion_to_augment=1,
                     spatial_rank=2)) 
    
#     reader.add_preprocessing_layers(RandomFlipLayer(
#                  flip_axes=(0,1))) 

    return reader

f, axes = plt.subplots(5,4,figsize=(15,15))
f.suptitle('The same input image, deformed under varying $\sigma$')

for i, axe in enumerate(axes): 
    std_sigma = 25 * i
    reader = create_image_reader(6, std_sigma)
    for ax in axe: 
        _, image_data, _ = reader(1)
        ax.imshow(image_data['cell'].squeeze(), cmap='gray')
        ax.imshow(image_data['label'].squeeze(), cmap='jet', alpha=0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Deformation Sigma = %i' % std_sigma)
