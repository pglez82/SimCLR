import torchvision


class Square(object):
    """Make image squared"""
    def __init__(self, fill=0):
        assert isinstance(fill, int)
        self.fill = fill

    def __call__(self, img):
        h, w = img.size[0],img.size[1]
        
        if h > w:
        return torchvision.transforms.functional.pad(img, (0,((h-w)//2)), self.fill, 'constant')
        else:
        return torchvision.transforms.functional.pad(img, (((w-h)//2),0),self.fill, 'constant')


class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.CenterCrop(size=size),
                torchvision.transforms.ToTensor()
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
