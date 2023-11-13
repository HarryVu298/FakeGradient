import torch
import copy
from torch.autograd import Variable

def deepfoolC(image, net, num_classes=10, overshoot=0.02, max_iter=50):
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    f_image = net(Variable(image[None, :, :, :], requires_grad=True)).data.flatten()
    I = f_image.argsort(descending=True)[:num_classes]

    label = I[0]
    Originallabel = f_image.argsort(descending=True)[:1000][0]

    input_shape = image.shape
    pert_image = copy.deepcopy(image)
    w = torch.zeros(input_shape)
    r_tot = torch.zeros(input_shape)

    loop_i = 0
    TheGradient = None

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net(x)
    k_i = label.item()

    while k_i == label and loop_i < max_iter:
        pert = float('inf')

        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.clone()
        if loop_i == 0:
            TheGradient = grad_orig.clone()

        for k in range(1, num_classes):
            x.grad.data.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.clone()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = abs(f_k) / w_k.norm()

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        r_i = (pert + 1e-4) * w / w.norm()
        r_tot += r_i

        if is_cuda:
            pert_image = image + (1 + overshoot) * r_tot.cuda()
        else:
            pert_image = image + (1 + overshoot) * r_tot

        x = Variable(pert_image, requires_grad=True)
        fs = net(x)
        k_i = fs.data.flatten().argmax().item()
        Protected = fs.data.flatten()[:1000].argmax().item()

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, Originallabel, Protected, pert_image, TheGradient
