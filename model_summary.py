from torchsummary import summary

from parse_args import parse_args, get_model

args = parse_args()
model = get_model(args)
summary(model, (3, args.image_size, args.image_size))
