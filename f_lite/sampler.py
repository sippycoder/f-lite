import numpy as np

from typing import Iterator, Optional
import math

import torch
from torch.utils.data import Dataset, Sampler


class StatefulDistributedSampler(Sampler):

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,  # True,
        seed: int = 0,
        drop_last: bool = False,
        # save_dir: str = "/jfs/data/sampler",
    ) -> None:
        """
        Though we don't use torch distributed, we reframe our shrading into distribued like sampler.

        Args:
            dataset: Dataset used for sampling.
            num_replicas (int, optional): Number of processes participating in
                distributed training. By default, :attr:`world_size` is retrieved from the
                current distributed group.
                * the alias for num_shard
            rank (int, optional): Rank of the current process within :attr:`num_replicas`.
                By default, :attr:`rank` is retrieved from the current distributed
                group.
                * the alias for shard_id
            shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
                indices.
            seed (int, optional): random seed used to shuffle the sampler if
                :attr:`shuffle=True`. This number should be identical across all
                processes in the distributed group. Default: ``0``.
            drop_last (bool, optional): if ``True``, then the sampler will drop the
                tail of the data to make it evenly divisible across the number of
                replicas. If ``False``, the sampler will add extra indices to make
                the data evenly divisible across the replicas. Default: ``False``.

            * shard_id and num_shard could be changed when loading and saving, we only save start idx
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        # * no need to consider sharding
        self.num_samples = len(self.dataset)
        self.total_size = self.num_samples
        self.shuffle = shuffle
        self.seed = seed

        # * most important variable, dataset traverse pointer
        self.start_index: int = 0
        self.shard_id = rank
        self.num_shard = num_replicas
        self.shuffle = shuffle
        # self.save_dir = save_dir

    def __iter__(self) -> Iterator:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).numpy()  # type: ignore[arg-type]
        else:
            indices = np.arange(len(self.dataset))
            # list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices = np.concatenate((indices, indices[:padding_size]))
            else:
                repeat_count = math.ceil(padding_size / len(indices))
                indices = np.concatenate(
                    (indices, np.tile(indices, repeat_count)[:padding_size])
                )
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size
        assert len(indices) == self.num_samples

        # * clip index list from start from state
        indices = indices[self.start_index * self.num_replicas + self.rank :]
        indices = indices[::self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index * self.num_replicas

    def reset(self, spefic_index: int = 0) -> None:

        # NOTE: we will have situations that we could mannually recover from a specific index
        #       Beside, I think set `shuffle = False` as default makes more sense according to our usage.
        self.start_index = 0 if spefic_index == 0 else spefic_index

        # NOTE: Previously, we set `shuffle = True`` by default, if so, it it better add the shuffle function.
        #       Otherwise we need to set it to False by default. In our situation, we don't need to shuffle frequently.
        # if shuffle is None:
        #     shuffle = self.shuffle
        # if shuffle:
        #     self.shuffle_indices()

    # def shuffle_indices(self) -> None:

    #     if self.shuffle:
    #         generator = torch.Generator()
    #         generator.manual_seed(self.seed)
    #         indices = torch.randperm(len(self.dataset), generator=generator).tolist()
    #         if hasattr(self.dataset, "indices"):
    #             # Update dataset's internal indices if supported
    #             self.dataset.indices = indices

    def state_dict(self, global_step) -> dict:
        local_step = (global_step * self.batch_size) % self.num_samples
        # * we shouldn't update start index during training
        # as it should only be init once the training start in self.__iter__
        return {"start_index": local_step}

    def load_state_dict(self, state_dict: dict) -> None:
        # self.__dict__.update(state_dict)
        # TODO: Please check here. We may explicit load the controlable attributes.
        self.start_index = state_dict.get("start_index", 0)
        self.seed = state_dict.get("seed", self.seed)
        self.shuffle = state_dict.get("shuffle", self.shuffle)

    # NOTE: we have already save TrainState which includes sampler in https://github.com/metauto-ai/Pollux/blob/main/apps/main/train.py#L543
    # def save_state(self, global_step: int) -> None:

    #     # NOTE: For safety, it is better we save the minimal info,
    #     # e.g., start index, seed, shuffle, global step for recovery.
    #     state_set_info = {
    #         "start_index": self.start_index,
    #         "seed": self.seed,
    #         "shuffle": self.shuffle,
    #         "global_step": global_step,
    #     }
    #     # Create a timestamped folder within the save_dir
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     timestamped_dir = os.path.join(self.save_dir, timestamp)
    #     os.makedirs(timestamped_dir, exist_ok=True)

    #     state_path = os.path.join(
    #         timestamped_dir, f"sampler_state_rank_{self.shard_id}.pth"
    #     )
    #     torch.save(state_set_info, state_path)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch



class ResolutionBucketSampler(torch.utils.data.BatchSampler):
    """Group images by resolution to ensure consistent resolution within a batch"""
    
    def __init__(
        self, 
        dataset, 
        batch_size, 
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle=True, 
        drop_last=True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        
        # Group images by aspect ratio
        self.buckets = dataset.aspect_ratio_buckets
        
        # State management for resumable training
        self.start_batch_index = 0
        
        print(f"Created {len(self.buckets)} aspect ratio buckets with keys: {list(self.buckets.keys())}")
        print(f"Distributed sampling: rank {rank}/{num_replicas}")
    
    def __iter__(self):
        # Use deterministic shuffling based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Create batches within each aspect ratio bucket
        batches = []
        for aspect, indices in self.buckets.items():
            # Shuffle indices within buckets if requested
            if self.shuffle:
                # Convert to tensor for deterministic shuffling
                indices_tensor = torch.tensor(indices)
                shuffled_indices = indices_tensor[torch.randperm(len(indices), generator=g)]
                indices = shuffled_indices.tolist()
            
            # Create complete batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:  # Handle incomplete batches
                    batches.append(batch)
        
        # Shuffle batch order if requested
        if self.shuffle:
            batch_indices = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_indices]
        
        # Apply distributed sampling if configured
        if self.num_replicas is not None and self.rank is not None:
            # Shard batches across replicas
            batches = batches[self.rank::self.num_replicas]
        
        # Apply state-based starting point for resumable training
        batches = batches[self.start_batch_index:]
        
        # Return batch list - note that batches are not flattened here
        return iter(batches)
    
    def __len__(self):
        # Calculate total number of batches
        if self.drop_last:
            total_batches = sum(len(indices) // self.batch_size for indices in self.buckets.values())
        else:
            total_batches = sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.buckets.values())
        
        # Account for distributed sampling
        if self.num_replicas is not None:
            # Each replica gets a subset of batches
            total_batches = (total_batches + self.num_replicas - 1) // self.num_replicas
        
        # Account for state-based starting point
        return max(0, total_batches - self.start_batch_index)
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.
        
        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
    
    def state_dict(self, global_step: int) -> dict:
        """
        Get the current state of the sampler for checkpointing.
        
        Args:
            global_step (int): Current global training step
            
        Returns:
            dict: State dictionary containing sampler state
        """
        # Calculate current batch index based on global step
        # This is an approximation - in practice you might want to track this more precisely
        local_batch_step = global_step % len(self)
        
        return {
            "start_batch_index": local_batch_step,
            "epoch": self.epoch,
            "seed": self.seed,
            "shuffle": self.shuffle,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the sampler state from a checkpoint.
        
        Args:
            state_dict (dict): State dictionary to load
        """
        self.start_batch_index = state_dict.get("start_batch_index", 0)
        self.epoch = state_dict.get("epoch", 0)
        self.seed = state_dict.get("seed", self.seed)
        self.shuffle = state_dict.get("shuffle", self.shuffle)
    
    def reset(self, specific_batch_index: int = 0) -> None:
        """
        Reset the sampler to a specific batch index.
        
        Args:
            specific_batch_index (int): Batch index to reset to
        """
        self.start_batch_index = specific_batch_index