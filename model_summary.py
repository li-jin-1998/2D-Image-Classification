from torchsummary import summary

from parse_args import parse_args, getModel

args = parse_args()
model = getModel(args)
summary(model, (3, args.image_size, args.image_size))
