import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data.separate import separate


class VOCSuperpixels(InMemoryDataset):
    r"""The VOCSuperpixels dataset which contains image superpixels and a semantic segmentation label
    for each node superpixel.
    
    Construction and Preparation:
    - The superpixels are extracted in a similar fashion as the MNIST and CIFAR10 superpixels. 
    - In VOCSuperpixels, the number of superpixel nodes <=500. (Note that it was <=75 for MNIST and
    <=150 for CIFAR10.)
    - The labeling of each superpixel node is done with the same value of the original pixel ground
    truth  that is on the mean coord of the superpixel node
    
    - Based on the SBD annotations from 11355 images taken from the PASCAL VOC 2011 dataset. Original
    source `here<https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal>`_.
    
    num_classes = 21
    ignore_label = 255

    color map
    0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow,
    11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train,
    20=tv/monitor
    
    Splitting:
    - In the original image dataset there are only train and val splitting.
    - For VOCSuperpixels, we maintain train, val and test splits where the train set is AS IS. The original
    val split of the image dataset is used to divide into new val and new test split that is eventually used
    in VOCSuperpixels. The policy for this val/test splitting is below.
    - Split total number of val graphs into 2 sets (val, test) with 50:50 using a stratified split proportionate
    to original distribution of data with respect to a meta label.
    - Each image is meta-labeled by majority voting of non-background grouth truth node labels. Then new val
    and new test is created with stratified sampling based on these meta-labels. This is done for preserving
    same distribution of node labels in both new val and new test
    - Therefore, the final train, val and test splits are correspondingly original train (8498), new val (1428)
    and new test (1429) splits.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): Option to select the graph construction format.
            If :obj: `"edge_wt_only_coord"`, the graphs are 8-nn graphs with the edge weights computed based on
            only spatial coordinates of superpixel nodes.
            If :obj: `"edge_wt_coord_feat"`, the graphs are 8-nn graphs with the edge weights computed based on
            combination of spatial coordinates and feature values of superpixel nodes.
            If :obj: `"edge_wt_region_boundary"`, the graphs region boundary graphs where two regions (i.e. 
            superpixel nodes) have an edge between them if they share a boundary in the original image.
            (default: :obj:`"edge_wt_region_boundary"`)
        slic_compactness (int, optional): Option to select compactness of slic that was used for superpixels
            (:obj:`10`, :obj:`30`). (default: :obj:`30`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    
    # The original LRGB Dropbox links are disabled; the LRGB authors mirror the
    # already-collated (data, slices) splits on Hugging Face.
    hf_url = ('https://huggingface.co/datasets/LRGB/'
              'voc_superpixels_{name}_{slic_compactness}/resolve/main/{split}.pt')

    def __init__(self, root, name='edge_wt_region_boundary', slic_compactness=30, split='train',
                 transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.slic_compactness = slic_compactness
        assert split in ['train', 'val', 'test']
        assert name in ['edge_wt_only_coord', 'edge_wt_coord_feat', 'edge_wt_region_boundary']
        assert slic_compactness in [10, 30]
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        from graphgps.loader.torch_io import torch_load
        self.data, self.slices = torch_load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def raw_dir(self):
        return osp.join(self.root,
                        'slic_compactness_' + str(self.slic_compactness),
                        self.name,
                        'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root,
                        'slic_compactness_' + str(self.slic_compactness),
                        self.name,
                        'processed')
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        for split in ['train', 'val', 'test']:
            # Raw files are moved into processed_dir, so skip splits already there.
            if osp.exists(osp.join(self.processed_dir, f'{split}.pt')):
                continue
            url = self.hf_url.format(name=self.name,
                                     slic_compactness=self.slic_compactness,
                                     split=split)
            download_url(url, self.raw_dir)

    def process(self):
        from graphgps.loader.torch_io import torch_load
        for split in ['train', 'val', 'test']:
            raw_path = osp.join(self.raw_dir, f'{split}.pt')
            processed_path = osp.join(self.processed_dir, f'{split}.pt')
            if osp.exists(processed_path):
                continue
            if self.pre_filter is None and self.pre_transform is None:
                shutil.move(raw_path, processed_path)
                continue

            data, slices = torch_load(raw_path)
            data_list = []
            for idx in range(len(slices['x']) - 1):
                graph = separate(cls=data.__class__, batch=data, idx=idx,
                                 slice_dict=slices, decrement=False)
                if self.pre_filter is not None and not self.pre_filter(graph):
                    continue
                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
                data_list.append(graph)
            torch.save(self.collate(data_list), processed_path)
            os.unlink(raw_path)
