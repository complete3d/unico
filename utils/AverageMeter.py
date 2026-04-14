
class AverageMeter(object):
    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values):
        if type(values).__name__ == 'list':
            for idx, v in enumerate(values):
                self._val[idx] = v
                self._sum[idx] += v
                self._count[idx] += 1
        elif type(values).__name__ == 'dict':
            for idx, v in enumerate(self.items):
                if v in values.keys():
                    self._val[idx] = values[v]
                    self._sum[idx] += values[v]
                    self._count[idx] += 1
        else:
            self._val[0] = values
            self._sum[0] += values
            self._count[0] += 1

    def val(self, idx=None, key=None):
        if idx is not None:
            return self._val[idx]
        elif key is not None:
            for idx, v in enumerate(self.items):
                if v == key:
                    return self._val[idx]
        else:
            return self._val[0] if self.items is None else [self._val[i] for i in range(self.n_items)]

    def count(self, idx=None):
        if idx is None:
            return self._count[0] if self.items is None else [self._count[i] for i in range(self.n_items)]
        else:
            return self._count[idx]

    def avg(self, idx=None, key=None):
        if idx is not None:
            return self._sum[idx] / self._count[idx]
        elif key is not None:
            for idx, v in enumerate(self.items):
                if v == key:
                    return self._sum[idx] / self._count[idx]
        else:
            return self._sum[0] / (self._count[0]+ 1e-6) if self.items is None else [
                self._sum[i] / (self._count[i]+ 1e-6) for i in range(self.n_items)
            ]
        
        
        
