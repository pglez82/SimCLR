import torchvision

class Square(object):
    """Make image squared"""
    def __init__(self, fill=0,size=224):
        assert isinstance(fill, int)
        assert isinstance(size, int)
        self.fill = fill
        self.size = size

    def __call__(self, img):

        w, h = img.size[0],img.size[1]
        
        if h > w:
            newsize = (self.size,round(w/(h/float(w))))
            totalpadding = newsize[0]-newsize[1]
            if (totalpadding % 2) != 0: 
                padding = (totalpadding//2,0,(totalpadding//2)+1,0)
            else:
                padding = (totalpadding//2,0,totalpadding//2,0)
            return torchvision.transform.functional.pad(torchvision.transform.functional.resize(img,newsize), padding, self.fill, 'constant')
        else:
            newsize = (round(h/(w/float(h))),self.size)
            totalpadding = newsize[1]-newsize[0]
            if (totalpadding % 2) != 0: 
                padding = (0,totalpadding//2,0,(totalpadding//2)+1)
            else:
                padding = (0,totalpadding//2,0,totalpadding//2)
            return torchvision.transform.functional.pad(torchvision.transform.functional.resize(img,newsize), padding, self.fill, 'constant')

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
