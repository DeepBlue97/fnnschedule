from fnnfunctor.loss.yolo import compute_loss


class Trainer():
    def __init__(self):
        self.start_epoch = 0
        self.max_epoch = None

        self.model = None

        self.device = None

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def before_train(self):
        self.model.train()

    def after_train(self):
        pass

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        imgs = imgs.to(self.device, non_blocking=True)  # torch.Size([8, 3, 416, 416])
        targets = targets.to(self.device)  # torch.Size([8, 6])

        outputs = self.model(imgs)

        loss, loss_components = compute_loss(outputs, targets, yolo_layer_anchors)
        loss.backward()
        

    def before_iter(self):
        pass

    def after_iter(self):
        pass
