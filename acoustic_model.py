import math
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from typing import List


class AcousticModel(torch.nn.Module):
    def __init__(self, input_dims, output_dims):
        super(AcousticModel, self).__init__()
        self.input_dim = input_dims
        self.const_component_dim = 100
        self.out_dim = output_dims

        # self.fixed_affine = torch.nn.Linear(in_features=220, out_features=220)
        self.preconditioned_affine_1 = torch.nn.Linear(220, 2000)
        self.preconditioned_affine_2 = torch.nn.Linear(500, 2000)
        self.preconditioned_affine_3 = torch.nn.Linear(500, 2000)
        self.preconditioned_affine_4 = torch.nn.Linear(250, 2000)
        self.preconditioned_affine_5 = torch.nn.Linear(250, self.out_dim)

        self.softmax = torch.nn.Softmax()
        self.norm = torch.nn.BatchNorm1d(num_features=250, affine=False)  # non-learnable

    def splice(self, input: torch.Tensor, context: List[int]):
        context = np.array(context)
        n_left_indices = len(context[context < 0])
        n_right_indices = len(context[context > 0])
        idx = torch.from_numpy(context)
        continuous_context_length = np.arange(context[0], context[-1] + 1, 1)

        selected_indices = idx + abs(idx[0])

        # Pad input tensor
        zeropad = torch.nn.ZeroPad1d((n_left_indices, n_right_indices))
        input_feats_padded = zeropad(input)

        # split tensor, splice and finally flatten
        out_repeated = input_feats_padded.unfold(dimension=0, size=len(continuous_context_length), step=1)
        out_spliced = out_repeated[:, selected_indices]
        out_spliced = out_spliced.flatten()
        return out_spliced

    # def sum_group_component(self, input_tensor, in_dim, out_dim):
    #     # todo: implement this function
    #     num_groups = out_dim
    #     assert in_dim % num_groups == 0
    #     group_size = in_dim / num_groups
    #     in_tensor = torch.tensor_split(input_tensor, num_groups)
    #     cpu_vec = np.zeros((num_groups, 2))
    #     reverse_cpu_vec = []
    #     curr_idx = 0
    #     for i in range(num_groups):
    #         cpu_vec[i][0] = curr_idx
    #         cpu_vec[i][1] = curr_idx + group_size
    #         curr_idx += group_size
    #         for j in range(int(cpu_vec[i][0]), int(cpu_vec[i][1])):
    #             reverse_cpu_vec.append(i)
    #
    #     indexes = cpu_vec
    #     reverse_indexes = np.array(reverse_cpu_vec)
    #
    #     out = input_tensor
    #     return out

    def forward(self, input_feats):
        """ block 1 """
        # component 0: SpliceComponent
        input_feats_to_splice = input_feats[:self.input_dim - self.const_component_dim]
        out_spliced = self.splice(input_feats_to_splice, context=[-1, 0, 1])
        out_spliced = torch.cat([out_spliced, input_feats[self.input_dim - self.const_component_dim:]])

        # component 1: FixedAffineComponent
        out_linear = self.fixed_affine(out_spliced)

        # component 2: AffineComponentPreconditionedOnline
        out_linear = self.preconditioned_affine_1(out_linear)

        # component 3: PnormComponent
        out_split = torch.tensor_split(out_linear, 250)  # sections input into 250 groups
        out_split = torch.stack(out_split, dim=0)
        out_norm = torch.norm(out_split, p=2, dim=1)  # elements are normed column-wise

        # component 4: NormalizeComponent
        out_tensor_norm = self.norm(out_norm)

        ''' block 2 '''
        # component 5: SpliceComponent
        out_spliced_2 = self.splice(out_tensor_norm, context=[-2, 1])

        # component 6: AffineComponentPreconditionedOnline
        out_linear_2 = self.preconditioned_affine_2(out_spliced_2)

        # component 7: PnormComponent
        out_split_2 = torch.tensor_split(out_linear_2, 250)  # sections input into 250 groups
        out_split_2 = torch.stack(out_split_2, dim=0)
        out_norm_2 = torch.norm(out_split_2, p=2, dim=1)  # elements are normed column-wise

        # component 8: NormalizeComponent
        out_tensor_norm_2 = self.norm(out_norm_2)

        ''' block 3 '''
        # component 9: SpliceComponent
        out_spliced_3 = self.splice(out_tensor_norm_2, context=[-4, 2])

        # component 10: AffineComponentPreconditionedOnline
        out_linear_3 = self.preconditioned_affine_3(out_spliced_3)

        # component 11: PnormComponent
        out_split_3 = torch.tensor_split(out_linear_3, 250)  # sections input into 250 groups
        out_split_3 = torch.stack(out_split_3, dim=0)
        out_norm_3 = torch.norm(out_split_3, p=2, dim=1)  # elements are normed column-wise

        # component 12: NormalizeComponent
        out_tensor_norm_3 = self.norm(out_norm_3)

        ''' block 4 '''
        # component 13: AffineComponentPreconditionedOnline
        out_linear_4 = self.preconditioned_affine_4(out_tensor_norm_3)

        # component 14: PnormComponent
        out_split_4 = torch.tensor_split(out_linear_4, 250)  # sections input into 250 groups
        out_split_4 = torch.stack(out_split_4, dim=0)
        out_norm_4 = torch.norm(out_split_4, p=2, dim=1)  # elements are normed column-wise

        # component 15: NormalizeComponent
        out_tensor_norm_4 = self.norm(out_norm_4)

        ''' block 5 '''
        # component 16: AffineComponentPreconditionedOnline
        out_linear_5 = self.preconditioned_affine_5(out_tensor_norm_4)

        # component 17: SoftmaxComponent
        out = self.softmax(out_linear_5)

        # # component 18: SumGroupComponent
        # out = self.sum_group_component(out_, in_dim=12000, out_dim=3373)

        return out


class SpeechData(IterableDataset):
    def __init__(self, start, end, pkl_data_file: str, transforms: List[str] = None):
        super(SpeechData, self).__init__()
        self.transforms = transforms
        self.start_idx = start
        self.end_idx = end
        self.ds = pkl.load(open(pkl_data_file, 'rb'))
        keys = list(self.ds)
        self.ds = [self.ds[keys[self.start_idx]], self.ds[keys[self.end_idx]]]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start_idx = self.start_idx
            iter_end_idx = self.end_idx
        else:
            load_per_worker = int(math.ceil(self.end_idx - self.start_idx) / float(worker_info.num_workers))
            worker_id = worker_info.id
            iter_start_idx = self.start_idx + worker_id * load_per_worker
            iter_end_idx = min(iter_start_idx + load_per_worker, self.end_idx)

        return iter(range(self.ds[iter_start_idx], self.ds[iter_end_idx]))

