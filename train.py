import argparse
import numpy.random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from timm.loss import LabelSmoothingCrossEntropy
from homura.vision.models.cifar_resnet import wrn28_10
from sam_bn import SAM_BN, ASAM_BN
import os
import time
from autoaugment import CIFAR10Policy

def load_cifar(data_loader, batch_size=256, num_workers=2, autoaugment=False, data_path = '/scratch/datasets/CIFAR100/'):
    if data_loader == CIFAR10:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    # Transforms
    if autoaugment:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=0),
                                              transforms.RandomHorizontalFlip(),
                                              CIFAR10Policy(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
    else:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # DataLoader
    train_set = data_loader(root=data_path, train=True, download=False, transform=train_transform)
    test_set = data_loader(root=data_path, train=False, download=False, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader

def train(args):
    state = {k: v for k, v in args._get_kwargs()}
    print(state)
    # Data Loader
    train_loader, test_loader = load_cifar(eval(args.dataset), args.m, autoaugment=args.autoaugment, data_path=args.data_path)
    num_classes = 100

    print('Creating Model...')
    # Model
    model = eval(args.model)(num_classes=num_classes).cuda()

    print('Model created.')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    print('Putting model on device...')
    model.to(device)
    print('On device.')
    # Minimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    if args.minimizer == 'SGD':
        minimizer = optimizer
    elif 'BN' in args.minimizer:
        minimizer = eval(args.minimizer)(optimizer, model, rho=args.rho, eta=args.eta, layerwise=args.layerwise,
                                         elementwise=args.elementwise, p=args.p, normalize_bias=args.normalize_bias,
                                         no_bn=args.no_bn, only_bn = args.only_bn)
    else:
        raise NotImplementedError
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer if args.minimizer=='SGD' else minimizer.optimizer, args.epochs)

    # Loss Functions
    if args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    print('Starting to train...')
    start_time = time.time()
    best_accuracy = 0.
    loss_best = 0.
    for epoch in range(args.epochs):
        epoch_start = time.time()
        # Train
        model.train()
        loss = 0.
        loss_adv = 0.
        accuracy = 0.
        cnt = 0.
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.type(torch.int64).to(device)
            predictions = model(inputs)
            batch_loss = criterion(predictions, targets)
            batch_loss.mean().backward()

            if args.minimizer=='SGD':
                minimizer.step()
                minimizer.zero_grad()
            else:
                # Ascent Step
                minimizer.ascent_step()
                # Descent Step
                predictions_adv = model(inputs)
                batch_loss_2 = criterion(predictions_adv, targets)
                batch_loss_2.mean().backward()
                minimizer.descent_step()

            with torch.no_grad():
                loss += batch_loss.sum().item()
                if args.minimizer == 'SGD':
                    loss_adv=0
                else:
                    loss_adv += batch_loss_2.sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
        if args.smoothing: # smoothing loss does reduce implicitly
            loss /= (idx+1)
            loss_adv /= (idx+1)
        else:
            loss /= cnt
            loss_adv /=cnt
        accuracy *= 100. / cnt
        print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}, Train loss adv. {loss_adv:8.5f}")
        scheduler.step()
        state['loss_train_adv_batch'] = batch_loss_2.mean().item()
        state['loss_train_adv_epoch'] = loss_adv
        state['loss_train'] = loss
        state['loss_train_batch'] = batch_loss.mean().item()
        state['accuracy_train'] = accuracy

        # Test
        model.eval()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.type(torch.int64).to(device)
                predictions = model(inputs)
                loss += criterion(predictions, targets).sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)
            if args.smoothing:  # smoothing loss does reduce implicitly
                loss /= (idx + 1)
            else:
                loss /= cnt
            accuracy *= 100. / cnt
        state['accuracy_test']=accuracy
        state['loss_test']=loss
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            # Save current best model
            torch.save(model.state_dict(),
                       os.path.join(args.save,
                                    'model_weights_best.pt'))
            loss_best = loss

        state['accuracy_test_best']=best_accuracy
        state['loss_test_best']=loss_best
        print(f"Epoch: {epoch}, Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}, Time: {time.time()-epoch_start}")
    end_time = time.time()
    state['runtime'] = end_time-start_time
    # Save last model
    torch.save(model.state_dict(),
               os.path.join(args.save,
                            + 'model_weights_last.pt'))

    # save final state
    filename = args.save + '/summary.csv'
    file_exists = os.path.isfile(filename)

    with open(filename, 'a') as f:
        if not file_exists:
            # write header
            f.write(','.join([str(i) for i in list(state.keys())]) + '\n')
        f.write(','.join([str(i) for i in list(state.values())]) + '\n')

    print(f"Best test accuracy: {best_accuracy}")
    print(f"Test accuracy at end of training: {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='CIFAR100', type=str, help="CIFAR10 or CIFAR100.")
    parser.add_argument("--data_path", default='/scratch/datasets/CIFAR100/', type=str, help="path to data root.")
    parser.add_argument("--model", default='wrn28_10', type=str, help="Name of model architecure")
    parser.add_argument("--minimizer", default='ASAM', type=str, help="ASAM_BN, SAM_BN or SGD")
    parser.add_argument("--p", default='2', type=str, choices=['2', 'infinity'])
    parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate.")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum.")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay factor.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing.")
    parser.add_argument("--rho", default=0.5, type=float, help="Rho for ASAM/SAM.")
    parser.add_argument("--layerwise", action='store_true', help="layerwise normalization for ASAM.")
    parser.add_argument("--elementwise", action='store_true', help="elementwise normalization for ASAM.")
    parser.add_argument("--autoaugment", action='store_true', help="apply autoaugment transformation.")
    parser.add_argument("--normalize_bias", action='store_true', help="apply ASAM also to bias params")
    parser.add_argument("--eta", default=0.0, type=float, help="Eta for ASAM.")
    parser.add_argument('--save', default='./snapshots', type=str, help='directory to save models in')
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument("--no_bn", action='store_true', help="perform ascent step without bn layer")
    parser.add_argument("--only_bn", action='store_true', help="perform ascent step only with bn layer")

    args = parser.parse_args()

    assert args.minimizer in ['SGD', 'SAM_BN', 'ASAM_BN'], \
        f"Invalid minimizer type. Please select ASAM or SAM"

    # set seed
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)
    # Make save directory
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)
    train(args)
