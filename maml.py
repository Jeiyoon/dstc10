"""
reimplementation and comments: Jeiyoon
"""
import torch
from torch import nn
from collections import OrderedDict

class MAML_Layer(nn.Module):
    def __init__(self, in_features, out_features):
        # super
        # https://velog.io/@gwkoo/%ED%81%B4%EB%9E%98%EC%8A%A4-%EC%83%81%EC%86%8D-%EB%B0%8F-super-%ED%95%A8%EC%88%98%EC%9D%98-%EC%97%AD%ED%95%A0
        super(MAML_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(in_features, 10)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(10, 10)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(10, out_features))
        ]))

    def forward(self, x):
        return self.fc(x)

    # model.parameterised():
    # 원래의 모델은 건드리지 않고 업데이트 버젼의 weight를 사용하기 위한 모듈
    # 업데이트 버젼으로 로스를 구하고, 얻어진 로스로 원래 모델을 업데이트 할 것이다
    def parameterised(self, x, weights):
        # self.model.parameterised(X, temp_weights)
        x = nn.functional.linear(x, weights[0], weights[1]) # (input, weight, bias)
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[2], weights[3])
        x - nn.functional.relu(x)
        x = nn.functional.linear(x, weights[4], weights[5])

        return x

class MAML():
    def __init__(self,
                 model,
                 train_tasks,
                 test_tasks,
                 inner_lr,
                 meta_lr,
                 K = 100, # K ???
                 inner_steps = 1,
                 tasks_per_meta_batch = 1000): # N ???

        # important objects
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.model = model

        # the maml weights we will be meta-optimising
        self.weights = list(model.parameters())

        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)

        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps
        self.tasks_per_meta_batch = tasks_per_meta_batch

        # metrics
        self.plot_every = 1
        self.print_every = 1
        self.train_losses = []
        self.test_losses = []

    # 여러번의 업데이트로 얻어진 weight로 얻어진 로스를 얻기 위한 내부의 loop
    # 업데이트를 몇 번 취할지는 자유로울 수 있으나 위의 구현에서는 1번만 수행하도록 제한
    def inner_loop(self, task):
        # reset inner model to current maml weights
        # torch.clone()
        # https://pytorch.org/docs/stable/generated/torch.clone.html
        temp_weights = [w.clone() for w in self.weights]

        # perform training an data sampled from task
        # torch.stack()
        # https://pytorch.org/docs/stable/generated/torch.stack.html
        X = torch.stack([b[0] for a in task for b in a])
        y = torch.stack([b[1] for a in task for b in a])

        for step in range(self.inner_steps):
            loss = self.criterion(self.model.parameterised(X, temp_weights), y) / len(y)

            # compute grad and update inner Loop weights
            # torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)
            # https://pytorch.org/docs/stable/generated/torch.autograd.grad.html
            # Inputs w.r.t. which the gradient will be returned (and not accumulated into .grad).
            # outputs (sequence of Tensor) – outputs of the differentiated function.
            # inputs: temp_weights
            # outputs: loss
            """
            grad_outputs (sequence of Tensor) – The “vector” in the Jacobian-vector product. 
            Usually gradients w.r.t. each output. None values can be specified for scalar Tensors or ones that don’t require grad. 
            If a None value would be acceptable for all grad_tensors, then this argument is optional. Default: None.
            """
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        # sample new data for meta-update and compute loss
        loss = self.criterion(self.model.parameterised(X, temp_weights), y) / len(y)

        return loss

    def main_loop(self, num_iterations):
        epoch_loss = 0

        for iteration in range(1, num_iterations + 1):
            # compute meta-loss
            train_loss = 0
            test_loss = 0

            train = [self.train_tasks[i] for i in self.train_tasks.keys()]
            train_loss += self.inner_loop(train)

            test = [self.test_tasks[i] for i in self.test_tasks.keys()]
            test_loss += self.inner_loop(test)

            # compute meta gradient of Loss w.r.t maml weights
            meta_grads = torch.autograd.grad(train_loss, self.weights)

            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g

            self.meta_optimiser.step()

            # Log metrics
            if iteration % 10 == 1:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(iteration,
                                                                                    num_iterations,
                                                                                    train_loss.item(),
                                                                                    test_loss.item()))

            self.train_losses.append(train_loss.item())
            self.test_losses.append(test_loss.item())

        print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(iteration,
                                                                            num_iterations,
                                                                            train_loss.item(),
                                                                            test_loss.item()))












