import torch
from torch.utils.data import Dataset
from collections import namedtuple
from torch import Tensor
from torch.utils.data import DataLoader

TrainingData1DPDE = namedtuple("TrainingData1DPDE", ["x_coor", "x_E", "y_true"])
TrainingData1DStressBC = namedtuple("TrainingData1DStressBC", ["x_coor", "x_E", "y_true"])


class Dataset(torch.utils.data.Dataset):
    def _repeat_tensor(self, tensor: Tensor, dim: tuple[int, ...]) -> Tensor:
        return tensor.repeat(dim)


class StretchedRod:
    def __init__(self,length: float) -> None:
        self.length = length

    def create_uniform_points(self, num_points: int):
        return torch.linspace(0.0, self.length, num_points, requires_grad=True).view(num_points, 1)
    
    def create_random_points(self, num_points: int) -> Tensor:
        return torch.rand((num_points, 1), requires_grad=True) * self.length

    def create_point_at_free_end(self) -> Tensor:
        return torch.full((1, 1), self.length, requires_grad=True)




class TrainingDataset1D(Dataset):
    def __init__(
            self,
            geometry: StretchedRod,
            traction: float,
            volume_force: float,
            min_youngs_modulus: float,
            max_youngs_modulus: float,
            num_points_pde: int,
            num_samples:int
            ):
        super().__init__()
        self._geometry = geometry
        self._traction = traction
        self._traction = traction
        self._volume_force = volume_force
        self._min_youngs_modulus = min_youngs_modulus
        self._max_youngs_modulus = max_youngs_modulus
        self._num_points_stress_bc = 1
        self._num_points_pde=num_points_pde
        self._num_samples=num_samples
        self._samples_pde: list[TrainingData1DPDE] = []
        self._samples_stress_bc: list[TrainingData1DStressBC] = []
        self._generate_samples()

    def _generate_samples(self) -> None:
        youngs_moduli_list = self._generate_uniform_youngs_modulus_list()
        for i in range(self._num_samples):
            youngs_modulus = youngs_moduli_list[i]
            self._add_pde_sample(youngs_modulus)
            self._add_stress_bc_sample(youngs_modulus)

    def _generate_uniform_youngs_modulus_list(self) -> list[float]:
        return torch.linspace(
            self._min_youngs_modulus, self._max_youngs_modulus, self._num_samples
        )

    def _add_pde_sample(self, youngs_modulus: float) -> None:
        shape = (self._num_points_pde, 1)
        x_coor = self._geometry.create_uniform_points(self._num_points_pde)
        x_E = self._repeat_tensor(torch.tensor([youngs_modulus]), shape)
        y_true = torch.zeros(shape)
        sample = TrainingData1DPDE(x_coor=x_coor.detach(),
                                   x_E=x_E.detach(),
                                   y_true=y_true.detach())
        self._samples_pde.append(sample)

    def _add_stress_bc_sample(self, youngs_modulus: float) -> None:
        shape = (self._num_points_stress_bc, 1)
        x_coor = self._geometry.create_point_at_free_end()
        x_E = self._repeat_tensor(torch.tensor([youngs_modulus]), shape)
        y_true = self._repeat_tensor(torch.tensor([self._traction]), shape)
        sample = TrainingData1DStressBC(
            x_coor=x_coor.detach(), x_E=x_E.detach(), y_true=y_true.detach()
        )
        self._samples_stress_bc.append(sample)


    def __len__(self):
        return self._num_samples
    
    def __getitem__(self, idx: int) -> tuple[TrainingData1DPDE, TrainingData1DStressBC]:
        sample_pde = self._samples_pde[idx]
        sample_stress_bc = self._samples_stress_bc[idx]
        return sample_pde, sample_stress_bc

    
def collate_training_data_1D(
    batch: list[tuple[TrainingData1DPDE, TrainingData1DStressBC]]
) -> tuple[TrainingData1DPDE,TrainingData1DStressBC]:
    x_coor_pde_batch = []
    x_E_pde_batch = []
    y_true_pde_batch = []
    x_coor_stress_bc_batch = []
    x_E_stress_bc_batch = []
    y_true_stress_bc_batch = []


    def append_to_pde_batch(sample_pde: TrainingData1DPDE) -> None:
        x_coor_pde_batch.append(sample_pde.x_coor)
        x_E_pde_batch.append(sample_pde.x_E)
        y_true_pde_batch.append(sample_pde.y_true)

    def append_to_stress_bc_batch(sample_stress_bc: TrainingData1DStressBC) -> None:
        x_coor_stress_bc_batch.append(sample_stress_bc.x_coor)
        x_E_stress_bc_batch.append(sample_stress_bc.x_E)
        y_true_stress_bc_batch.append(sample_stress_bc.y_true)


    for sample_pde, sample_stress_bc in batch:
        append_to_pde_batch(sample_pde)
        append_to_stress_bc_batch(sample_stress_bc)

    batch_pde = TrainingData1DPDE(
        x_coor=torch.concat(x_coor_pde_batch, dim=0),
        x_E=torch.concat(x_E_pde_batch, dim=0),
        y_true=torch.concat(y_true_pde_batch, dim=0),
    )

    batch_stress_bc = TrainingData1DStressBC(
        x_coor=torch.concat(x_coor_stress_bc_batch, dim=0),
        x_E=torch.concat(x_E_stress_bc_batch, dim=0),
        y_true=torch.concat(y_true_stress_bc_batch, dim=0),
    )


    return batch_pde, batch_stress_bc

    
def create_training_dataset_1D(
            length: float,
            traction: float,
            volume_force: float,
            min_youngs_modulus: float,
            max_youngs_modulus: float,
            num_points_pde: int,
            num_samples: int
):
    
    geometry = StretchedRod(length=length)
    return TrainingDataset1D(        
        geometry=geometry,
        traction=traction,
        volume_force=volume_force,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        num_points_pde=num_points_pde,
        num_samples=num_samples,
)


'''torch.manual_seed(1)
#n_points = 50
youngs_modulus = 195.0
force = 100.0
area = 20.0
sig_applied = force/area
Vol_force = 0.0
length = 1.0
num_samples_train = 10
num_points_pde = 5
batch_size_train = num_samples_train
num_epochs = 50
traction = 5.0
volume_force = 0.0
min_youngs_modulus = 180.0
max_youngs_modulus = 240.0
displacement_left = 0.0

print("Create training data ...")
train_dataset = create_training_dataset_1D(length=length,
        traction=traction,
        volume_force=volume_force,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        num_points_pde=num_points_pde,
        num_samples=num_samples_train
        )

train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_train,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_training_data_1D,
    )

train_batches = iter(train_dataloader)

for batch_pde, batch_stress_bc in train_batches:
    print(batch_pde)
    print(batch_stress_bc)'''