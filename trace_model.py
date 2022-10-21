from predict import load_model
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch


# module = load_model().net
# module = module.to("cpu")
example = torch.rand((1, 3, 64, 64))

# trace and optimize
# traced_module = torch.jit.trace(module, example)
# optimized_model = optimize_for_mobile(traced_module)
#
# optimized_model.save('mobile_module.pt')

model = torch.jit.load('mobile_module.pt')
pred_tensor = model(example)
print(pred_tensor)
