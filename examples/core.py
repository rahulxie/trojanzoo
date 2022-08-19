def optimize_mark(self, label: int,
                  loader: Iterable = None,
                  logger_header: str = '',
                  verbose: bool = True,
                  **kwargs) -> tuple[torch.Tensor, float]:
    r"""
    Args:
        label (int): The class label to optimize.
        loader (collections.abc.Iterable):
            Data loader to optimize trigger.
            Defaults to ``self.dataset.loader['train']``.
        logger_header (str): Header string of logger.
            Defaults to ``''``.
        verbose (bool): Whether to use logger for output.
            Defaults to ``True``.
        **kwargs: Keyword arguments passed to :meth:`loss()`.

    Returns:
        (torch.Tensor, torch.Tensor):
            Optimized mark tensor with shape ``(C + 1, H, W)``
            and loss tensor.
    """
    self.model._model.defense = True
    atanh_mark = torch.randn_like(self.attack.mark.mark, requires_grad=True)
    optimizer = optim.Adam([atanh_mark], lr=self.defense_remask_lr, betas=(0.5, 0.9))

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=self.defense_remask_epoch)
    optimizer.zero_grad()
    loader = loader or self.dataset.loader['train']

    # best optimization results
    norm_best: float = float('inf')
    mark_best: torch.Tensor = None
    loss_best: float = None

    logger = MetricLogger(indent=4)
    logger.create_meters(loss='{last_value:.3f}',
                         acc='{last_value:.3f}',
                         norm='{last_value:.3f}',
                         entropy='{last_value:.3f}', )
    batch_logger = MetricLogger()
    logger.create_meters(loss=None, acc=None, entropy=None)

    iterator = range(self.defense_remask_epoch)
    if verbose:
        iterator = logger.log_every(iterator, header=logger_header)
    for i in iterator:
        batch_logger.reset()
        for data in loader:
            self.attack.mark.mark = tanh_func(atanh_mark)  # (c+1, h, w)
            _input, _label = self.model.get_data(data)
            trigger_input = self.attack.add_mark(_input)
            trigger_label = label * torch.ones_like(_label)
            trigger_output = self.model(trigger_input)

            batch_acc = trigger_label.eq(trigger_output.argmax(1)).float().mean()
            batch_entropy = self.loss(self.attack.mark.mark, _input, _label,
                                      target=label,
                                      trigger_output=trigger_output,
                                      **kwargs)
            batch_norm: torch.Tensor = self.attack.mark.mark[-1].norm(p=1)
            batch_loss = batch_entropy + self.cost * batch_norm

            # gradient estimation
            if hasattr(self, 'est'):
                device = trigger_input.device
                with torch.no_grad():
                    q = 10
                    if hasattr(self, 'q'):
                        q = self.q
                    trigger_input_noise = torch.tensor([])
                    mark_norm_noise, regularization_noise = 0, 0
                    for j in range(q):
                        atanh_mark_clone = atanh_mark.clone()
                        u = torch.normal(0, 1, size=atanh_mark_clone.size())
                        u_norm = torch.norm(u, p=2, dim=[1, 2]).reshape(atanh_mark_clone.size()[0], 1, 1).expand(
                            atanh_mark_clone.size())
                        u = torch.div(u, u_norm).to(device)
                        atanh_mark_clone += u
                        self.attack.mark.mark = tanh_func(atanh_mark_clone)
                        mark_norm_noise += self.attack.mark.mark[-1].norm(p=1)
                        regularization_noise += self.regularization_loss(self.attack.mark.mark)
                        trigger_input_noise = torch.cat([trigger_input_noise, self.attack.add_mark(_input)], dim=0)
                    trigger_output_noise = self.model(trigger_input_noise)
                    trigger_label_copy = torch.cat([trigger_label for i in range(10)], dim=0)
                    # self.model.loss CrossEntropy
                    grad = (self.model.loss(trigger_input_noise, trigger_label_copy, trigger_output_noise)
                            - self.model.loss(trigger_input, trigger_label, trigger_output)
                            + self.cost * (mark_norm_noise / q - batch_norm)
                            + regularization_noise / q
                            - self.regularization_loss(tanh_func(atanh_mark))) \
                           * u
                self.attack.mark.mark = tanh_func(atanh_mark)
                loss = torch.sum(atanh_mark * grad)
                loss.backward()
            else:
                batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_size = _label.size(0)
            batch_logger.update(n=batch_size,
                                loss=batch_loss.item(),
                                acc=batch_acc.item(),
                                entropy=batch_entropy.item())
        lr_scheduler.step()
        self.attack.mark.mark = tanh_func(atanh_mark)  # (c+1, h, w)

        # if (i+1) % 5 == 0:
        #     self.cost *= 2

        # check to save best mask or not
        loss = batch_logger.meters['loss'].global_avg
        acc = batch_logger.meters['acc'].global_avg
        norm = float(self.attack.mark.mark[-1].norm(p=1))
        entropy = batch_logger.meters['entropy'].global_avg
        if norm < norm_best:
            mark_best = self.attack.mark.mark.detach().clone()
            loss_best = loss
            logger.update(loss=loss, acc=acc, norm=norm, entropy=entropy)

        if self.check_early_stop(loss=loss, acc=acc, norm=norm, entropy=entropy):
            print('early stop')
            break
    atanh_mark.requires_grad_(False)
    self.attack.mark.mark = mark_best
    return mark_best, loss_best
