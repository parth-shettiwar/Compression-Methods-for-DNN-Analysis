import torchvision.datasets as datasets
import torchvision.transforms as T


def get_dataset(dataset, normalize=True):
	if dataset == 'mnist':
		transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]) if normalize else T.ToTensor()
		train_dataset = datasets.MNIST('./dataset/mnist', train=True, download=True, transform=transform)
		test_dataset = datasets.MNIST('./dataset/mnist', train=False, download=True, transform=transform)

	elif dataset == 'cifar10':
		transform_train = T.Compose([
			T.RandomCrop(32, padding=4),
			T.RandomHorizontalFlip(),
			T.ToTensor(),
			T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))],
		)
		transform_test = T.Compose([
			T.ToTensor(),
			T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))],
		)
		train_dataset = datasets.CIFAR10('./dataset/cifar10', train=True, download=True, transform=transform_train)
		test_dataset = datasets.CIFAR10('./dataset/cifar10', train=False, download=True, transform=transform_test)

	

	return train_dataset, test_dataset


