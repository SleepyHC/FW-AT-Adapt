from pathlib import Path

import autoattack as aa
import argparse
import yaml
from AdversarialTrainer import AdversarialTrainer
import DataLoad as DL
from advertorch.attacks import LinfPGDAttack, L2PGDAttack, FGSM
import torch.nn as nn
import torch

LOADERS = {"cifar10": DL.get_loaders_cifar10, "cifar100": DL.get_loaders_cifar100}

MODELS = {
    "cifar10": "cifar10_resnet18_baseline_nat_acc_94.pt",
    "cifar100": "cifar100_resnet18_baseline_nat_acc_76p3.pt",
}

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_best_checkpoint_simple(cd):
    bpath = list(Path(cd).rglob("checkpoint__best.pt"))
    return bpath[0]

def accu(model, dataloader):
    model1 = model.model.eval()
    model1.to(device)
    acc = 0
    for input, target in dataloader:
        input = input.to(device)
        target = target.to(device)
        input = adversary_8.perturb(input,target)
        o = model1(input)
        acc += (o.argmax(dim=1).long() == target).float().mean()
    print ("robust acc is ",acc / len(dataloader))
    return acc / len(dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--ds", default="cifar10")
    parser.add_argument("--data_path", default="./data", help="Path to data directory.")
    parser.add_argument("--break_at", default=-1)

    args = parser.parse_args()

    # load data
    loader_fn = LOADERS[args.ds]
    _, val_loader = loader_fn(
        data_path=args.data_path,
        batch_size_train=16,
        batch_size_val=2048,
        num_workers=20,
    )

    exp_dir = Path(args.exp_dir)
    logout_path = exp_dir.joinpath(f"autoattack_results_{args.tag}.txt")

    #  with exp_dir.joinpath("hparams.yaml").open("r") as f:
    #     eps = yaml.safe_load(f)["epsilon"]
    eps=8/255
    # Load model
    model_path = (
        get_best_checkpoint_simple(exp_dir)
        if args.model_path is None
        else args.model_path
    )

    num_classes = 100 if args.ds == "cifar100" else 10
    model = AdversarialTrainer("resnet18", topdir=exp_dir, num_classes=num_classes)
    model.load_model(model_path)

    model.model.eval()

    def forward_pass(x):
        return model.forward(x, inplace=False)

    # adversary = aa.AutoAttack(forward_pass, norm="Linf", eps=eps, log_path=logout_path)

    # Run the evaluation
    # num_cor = 0
    # num_total = 0.0
    # num_b = 0
    # for batch in val_loader:
    #     num_b += 1
    #     if num_b == args.break_at:
    #         break
    #     x0, y0 = batch
    #     _, yadv = adversary.run_standard_evaluation(x0, y0, return_labels=True, bs=1024)

    #     cor = (yadv.cpu() == y0).sum().item()
    #     num_cor += cor
    #     num_total += y0.shape[0]

    # res = {"aa_acc": 100 * (num_cor / num_total)}

    eval_args = {
        "mode": "pgd",
        "K": 50,
        "epsilon": eps,
        "adv_norm": "Linf",
        "adv": True,
    }
    print('pgd-50')
    eval_res = model.evaluate(val_loader, **eval_args)
    eps=8/255
    eps_iter=2/255
    adversary_8 = LinfPGDAttack(
                        model.model, loss_fn=nn.CrossEntropyLoss(), eps=eps, nb_iter=20, eps_iter=eps_iter,
                        rand_init=1, clip_min=0.0, clip_max=1.0, targeted=False
                    )
    accu(model,val_loader)
    adversary_8 = LinfPGDAttack(
                        model.model, loss_fn=nn.CrossEntropyLoss(), eps=eps, nb_iter=50, eps_iter=eps_iter,
                        rand_init=1, clip_min=0.0, clip_max=1.0, targeted=False
                    )
    accu(model,val_loader)
    adversary_8 = FGSM(
            model.model, loss_fn=nn.CrossEntropyLoss(), eps=eps,
            clip_min=0.0, clip_max=1.0, targeted=False
        )
    print('FGSM:')
    accu(model,val_loader)
    res["pgd50_acc"] = eval_res["acc"]
    with exp_dir.joinpath("autoattack_and_pgd50_value.yaml").open("w") as f:
        yaml.dump(res, f)
