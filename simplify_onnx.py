from onnxsim import simplify

import onnx
from parse_args import parse_args

args = parse_args()

onnx_file_name = "./weights/{}_best_model_ssr.onnx".format(args.arch)

simplify_onnx_file_name = "./weights/{}_best_model_ssr_simplify.onnx".format(args.arch)

onnx_model = onnx.load(onnx_file_name)
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, simplify_onnx_file_name)
print('finished exporting onnx')
