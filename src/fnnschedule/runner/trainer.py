import os

from fnnschedule import *


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            # self.next_input, self.next_target, _, _ = next(self.loader)
            # _, self.next_input, self.next_target = next(self.loader)
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


class Trainer():
    def __init__(self, config):
        # self.iter_count = 0
        self.start_epoch = 1
        self.max_epoch = config.schedule['max_epoch']
        self.log_interval = config.schedule['log_interval']
        self.save_interval = config.schedule['save_interval']
        self.output_dir = config.output_dir


        # module_fnnmodel = importlib.import_module('fnnmodel')

        # # 获取类对象
        # class_model = getattr(module_fnnmodel, config.model['type'])

        # 实例化对象
        # self.model = class_model(config.model)
        # self.model = class_model(**config.model)
        self.model = dict2cls(config.model)

        # self.model.get_model(config.model['module']['type'], **config.model['module']['kwargs'])
        # self.model = type('YOLOv3')()

        # self.optimizer = self.model.get_optimizer()
        self.optimizer = self.model.optimizer

        # self.device = 

        # self.train_dataset = dict2cls(config.train_dataloader['dataset'])
        # self.val_dataset = dict2cls(config.val_dataloader['dataset'])


        # AUGMENTATION_TRANSFORMS = transforms.Compose([
        #     AbsoluteLabels(),
        #     DefaultAug(),
        #     PadSquare(),
        #     RelativeLabels(),
        #     ToTensor(),
        # ])

        # self.train_loader = torch.utils.data.DataLoader(
        #     self.train_dataset,
        #     batch_size=32,
        #     shuffle=True,
        #     num_workers=2,
        #     pin_memory=True,
        #     # collate_fn=self.train_dataset.collate_fn,
        #     # worker_init_fn=worker_seed_set
        # )
        # self.val_loader = torch.utils.data.DataLoader(
        #     self.val_dataset,
        #     batch_size=32,
        #     shuffle=False,
        #     num_workers=2,
        #     pin_memory=True,
        #     # collate_fn=self.val_dataset.collate_fn,
        #     # worker_init_fn=worker_seed_set
        # )
        self.train_loader = dict2cls(config.train_dataloader, recursive=True)
        self.val_loader = dict2cls(config.val_dataloader, recursive=True)


        # for im, (_a, _b, _c) in self.train_loader:
        #     print(im.shape)
        #     print(_b.shape)
        #     print()

        # for i, imgs in enumerate(self.train_loader):
        #     print(imgs)
        #     print(imgs)

            # print(_1)


        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self):
        return self.model.device

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def val(self):
        self.before_val()
        try:
            self.val_in_epoch()
        except Exception:
            raise
        finally:
            self.after_val()

    def val_in_epoch(self):
        for self.iter_val, (imgs, targets) in enumerate(self.val_loader):
            self.val_one_iter(imgs, targets)

    def val_one_iter(self, imgs, targets):
        # def train_one_iter(self):
            # imgs, targets = self.prefetcher.next()
            
            imgs = imgs.to(self.device, non_blocking=True)  # torch.Size([8, 3, 416, 416])
            # targets = targets.to(self.device)  # torch.Size([8, 6])

            outputs = self.model.forward(imgs)

            loss = self.model.computer_loss(outputs, targets)
            # self.loss_str = self.model.log_loss()

    
    def before_val(self):
        self.model.model.eval()

    def before_train(self):
        # torch.cuda.set_device('cuda')
        # self.model.model.to(self.device)
        self.model.model.train()

        self.max_iter = len(self.train_loader)

        # self.prefetcher = DataPrefetcher(self.train_loader)

    def after_train(self):
        # self.save_model()
        # self.log()
        pass

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch+1):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def before_epoch(self):
        pass

    def save_model(self):
        os.makedirs(self.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.output_dir, f'epoch_{self.epoch}.pth')
        self.model.save_model(checkpoint_path)
        print(f'save to: {checkpoint_path}')

    def after_epoch(self):
        if self.epoch % self.save_interval == 0:
            self.save_model()
        self.log()

    def train_in_iter(self):
        # for self.iter in range(self.max_iter):
        # for i, (imgs, targets) in self.train_loader:
        for self.iter, (imgs, targets) in enumerate(self.train_loader):
            self.before_iter()
            self.train_one_iter(imgs, targets)
            # self.train_one_iter()
            self.after_iter()

    def train_one_iter(self, imgs, targets):
    # def train_one_iter(self):
        # imgs, targets = self.prefetcher.next()
        
        imgs = imgs.to(self.device, non_blocking=True)  # torch.Size([8, 3, 416, 416])
        # targets = targets.to(self.device)  # torch.Size([8, 6])

        outputs = self.model.forward(imgs)

        loss = self.model.computer_loss(outputs, targets)
        # print(f'loss: {loss}' )
        self.loss_str = self.model.log_loss()



        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def before_iter(self):
        pass

    def log(self):
        print(f'epoch:{self.epoch}/{self.max_epoch} iter:{self.iter}/{self.max_iter} loss: {self.loss_str}')

    def after_iter(self):
        # self.iter_count += 1
        # if self.iter % self.log_interval == 0:
        #     print(f'epoch:{self.epoch}/{self.max_epoch} iter:{self.iter}/{self.max_iter}')
        if self.iter % self.log_interval == 0:
            self.log()
