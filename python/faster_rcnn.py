from pathlib import Path

from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torch import optim

if __name__ == '__main__':
    model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=2).to("cuda")
    root = Path("/home/vstupakov/DATA/gen3_projects/dataset #1/FasterRCNNv2/training/cocolike_dataset")

    dataset_train = CocoDetection(root=str(root / "train"), annFile=str(root / "annotations/instanses_train.json"))
    loader = DataLoader(dataset_train, batch_size=4, num_workers=4, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    epoches = 10
    for epoch in range(epoches):
        model.train()
        for i, (data, target) in enumerate(loader):
            data, target = data.to("cuda"), target.to("cuda")
            optimizer.zero_grad()

            out = model(data)




