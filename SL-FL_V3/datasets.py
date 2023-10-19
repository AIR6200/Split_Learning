import numpy as np
import torch 
from torchvision import datasets, transforms
import pandas as pd
import torch.utils.data as Data

def get_dataset(dir, name):

	if name=='mnist':
		train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
		eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
		
	elif name=='cifar':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		
		#modify the "dir"=>dir
		train_dataset = datasets.CIFAR10("E:/githubFile/federated-learning/data/cifar", train=True, download=True,
										transform=transform_train)
		eval_dataset = datasets.CIFAR10("E:/githubFile/federated-learning/data/cifar", train=False, transform=transform_test)

	elif name == 'FMNIST':
		data_transformer = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,))
		])
		train_dataset = datasets.FashionMNIST('data',train=True, download=True,transform=data_transformer)
		eval_dataset = datasets.FashionMNIST('data', train=False, transform=data_transformer)

	return train_dataset, eval_dataset