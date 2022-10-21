from predict import load_model
import torch
import coremltools as ct
import json

module = load_model().net
input_batch = torch.rand((1, 3, 64, 64))

#trace
traced_module = torch.jit.trace(module, input_batch)

#convert to coreml model
mlmodel = ct.convert(
    traced_module,
    inputs=[ct.TensorType(name="input", shape=input_batch.shape)],
)

# Save the model without new metadata
mlmodel.save("SegmentationModel_no_metadata.mlmodel")

# Load the saved model
# mlmodel = ct.models.MLModel("SegmentationModel_no_metadata.mlmodel")
#
# # Add new metadata for preview in Xcode
# labels_json = {"labels": ["background", "aeroplane", "bicycle", "bird", "board", "bottle", "bus", "car", "cat", "chair", "cow", "diningTable", "dog", "horse", "motorbike", "person", "pottedPlant", "sheep", "sofa", "train", "tvOrMonitor"]}
#
# mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
# mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)
#
# mlmodel.save("SegmentationModel_with_metadata.mlmodel")