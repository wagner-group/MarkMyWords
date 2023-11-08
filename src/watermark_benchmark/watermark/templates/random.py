from abc import ABC, abstractmethod
import torch
import random
import hash_cpp


class Randomness(ABC):

    @abstractmethod
    def __init__(self, secret_key, devices, vocab_size):
        self.secret_key = secret_key

        if type(devices) == list:
            self.devices = devices
            self.device = devices[0]
        else:
            self.device = devices
            self.devices = [devices]

        self.vocab_size = vocab_size
        self.state = None
        self.reset()
        self.set_permutation()

        self.generator = torch.Generator("cpu")

    def reset(self):
        l = 1 if type(self.secret_key) != list else len(self.secret_key)
        self.state = torch.zeros((l,)).long()

    def normalize_previous_values(self, previous_values):
        # Normalize input value
        if type(previous_values) != torch.Tensor:
            max_len = max(len(p) for p in previous_values) 
            previous_values = [[1 for _ in range(max_len - len(p))] + p for p in previous_values]
            previous_values = torch.tensor(previous_values)

        if len(previous_values.shape) == 1:
            previous_values = previous_values.unsqueeze(0)

        return previous_values

    def get_seed(self, previous_values, ids):
        self.state[ids] += 1

    @abstractmethod
    def rand_index(self, seeds, index, device=None):
        pass

    @abstractmethod
    def rand_range(self, seeds, length, device=None):
        pass

    def get_secret(self, offset):
        return self.secret_key[offset] if type(self.secret_key) == list else self.secret_key

    def set_permutation(self):
        if self.vocab_size < 2:
            return

        if type(self.secret_key) == list:
            shuf = [list(range(self.vocab_size)) for _ in range(len(self.secret_key))]
            for idx, key in enumerate(self.secret_key):
                random.Random(key).shuffle(shuf[idx])
            permutation= torch.tensor(shuf)
        else:
            shuf = list(range(self.vocab_size))
            random.Random(self.secret_key).shuffle(shuf)
            permutation= torch.tensor(shuf).unsqueeze(0)

        inv_permutation = torch.zeros_like(permutation)
        indices = torch.arange(permutation.shape[0]).repeat(permutation.shape[1],1).t()
        indices = torch.cat((indices.unsqueeze(2), permutation.unsqueeze(2)), dim=2)
        inv_permutation[indices[:,:,0], indices[:,:,1]] = torch.arange(self.vocab_size).repeat(permutation.shape[0],1)
        
        self.permutation = {device: permutation.to(device) for device in self.devices}
        self.inv_permutation = {device: inv_permutation.to(device) for device in self.devices}

    def get_permutation(self, device, inv=False):
        if device not in self.permutation:
            if type(device) == torch.device and device.index in self.permutation:
                device = device.index
            else:
                print("Device not initialized for random number generator. The sampling procedure is occuring on device {}, while only {} are available. Copying over".format(device, self.devices))
                self.permutation[device] = self.permutation[self.device].to(device)
                self.inv_permutation[device] = self.inv_permutation[self.device].to(device)
        if inv:
            return self.inv_permutation[device]
        else:
            return self.permutation[device]


    def green_list(self, seeds, gamma,inv=False):
        gl_size = int(gamma*self.vocab_size)
        permutation = torch.cat( tuple(torch.randperm(self.vocab_size, generator=self.generator.manual_seed(int(h.item() * 2147483647))).unsqueeze(0) for h in seeds) )
        if not inv:
            return permutation[:, :gl_size]
        else:
            permutation = permutation.to(self.device)
            inv_permutation = torch.zeros_like(permutation)
            indices = torch.arange(permutation.shape[0], device=self.device).repeat(permutation.shape[1],1).t()
            indices = torch.cat((indices.unsqueeze(2), permutation.unsqueeze(2)), dim=2)
            inv_permutation[indices[:,:,0], indices[:,:,1]] = torch.arange(self.vocab_size,device=self.device).repeat(permutation.shape[0],1)
            return inv_permutation <= gl_size


class EmbeddedRandomness(Randomness):

    def __init__(self, secret_key, device, vocab_size, hash_len, min_hash):
        super().__init__(secret_key, device, vocab_size)
        self.hash_len = hash_len
        self.min_hash = min_hash


    def get_seed(self, previous_values, ids=None):
        previous_values = self.normalize_previous_values(previous_values)
        N, _ = previous_values.shape
        if ids is None:
            ids = [0 for _ in range(N)]

        if not self.hash_len:
            tmp = [[] for _ in range(previous_values.shape[0])]
        else:
            tmp = [[v.item() for v in prev[-self.hash_len:]] for prev in previous_values]
            tmp = [[-1 for _ in range(self.hash_len - len(value))] + value for value in tmp]

        if self.min_hash:
            h = [str( round(min(hash_cpp.index_hash(["{}SEED{}".format(t, self.get_secret(ids[k]))], 0).cpu().item() for t in (tmp[k] if len(tmp[k]) else [0]) ), 8)) for k in range(N)]
        else:
            tmp = ["_".join(str(i) for i in t) if len(t) else "" for t in tmp]
            h = ["{}SEED{}".format(t, self.get_secret(ids[k])) for k,t in enumerate(tmp)]

        super().get_seed(previous_values, ids)
        return h


    def rand_range(self, seeds, length, device=None):
        if length == 0:
            length = self.vocab_size
        return hash_cpp.all_index_hash(seeds, torch.zeros((len(seeds), length), dtype=torch.float32).to(self.device if device is None else device))

    
    def rand_index(self, seeds, index, device=None):
        return hash_cpp.index_hash(seeds, index).to(self.device  if device is None else device)


class ExternalRandomness(Randomness):

    def __init__(self, secret_key, device, vocab_size, key_len=512, random_size = None):
        self.key_len = key_len
        super().__init__(secret_key, device, vocab_size)

        self.rng = [random.Random(self.secret_key)] if type(self.secret_key) != list else [random.Random(key) for key in self.secret_key]

        if random_size is None:
            random_size = vocab_size

        self.random_size = random_size
        
        self.xi = torch.tensor([[r.random() for _ in range(self.key_len*self.random_size)] for r in self.rng], dtype=torch.float32).reshape(len(self.rng), self.key_len, self.random_size)


    def reset(self):
        super().reset()
        l = 1 if type(self.secret_key) != list else len(self.secret_key)
        self.shift = torch.randint(self.key_len, (l,))
        #self.shift = torch.zeros((l,)).long().to(self.device)


    def get_seed(self, previous_values, ids=None):
        previous_values = self.normalize_previous_values(previous_values)
        N, _ = previous_values.shape
        if ids is None:
            ids = torch.zeros((N,)).long()
        elif type(ids) != torch.Tensor:
            ids = torch.Tensor(ids).long()
        super().get_seed(previous_values, ids)

        rtn = torch.cat((ids.unsqueeze(0), ((self.shift[ids] + self.state[ids] - 1)%self.key_len).unsqueeze(0)), axis=0).t()
        return rtn
        

    def rand_range(self, index, length, device=None):
        if length:
            return self.xi[index[:,0], index[:,1], :length].to(self.device if device is None else device)
        else:
            return self.xi[index[:,0], index[:,1], :].to(self.device if device is None else device)

    
    def rand_index(self, index, token_index, device=None):
        return self.xi[index[:,0], index[:,1], token_index].to(self.device if device is None else device)



