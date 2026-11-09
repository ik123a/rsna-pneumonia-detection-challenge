"""Execute the real loader with fakes; no torch/model/dataset downloads."""
import ast
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

class InferenceLoadingTests(unittest.TestCase):
    def test_nonpretrained_factory_disables_backbone_download(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'model.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_faster_rcnn_model')
        constructor = Mock(return_value=types.SimpleNamespace(roi_heads=types.SimpleNamespace(box_predictor=types.SimpleNamespace(cls_score=types.SimpleNamespace(in_features=256)))))
        namespace = dict(torchvision=types.SimpleNamespace(), fasterrcnn_resnet50_fpn=constructor, FastRCNNPredictor=Mock())
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'model.py', 'exec'), namespace)
        namespace['get_faster_rcnn_model'](pretrained=False)
        self.assertIn('weights_backbone', constructor.call_args.kwargs)
        self.assertIsNone(constructor.call_args.kwargs['weights_backbone'])

    def test_missing_checkpoint_never_constructs_model(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_inference_model')
        factory = Mock(side_effect=AssertionError('must not create an untrained model'))
        namespace = dict(_loaded_model=None, _loaded_model_device=None,
            model_lock=threading.Lock(),
            torch=types.SimpleNamespace(device=lambda value: value, cuda=types.SimpleNamespace(is_available=lambda: False)),
            config=types.SimpleNamespace(MODEL_SAVE_PATH='absent-checkpoint.pth'),
            os=types.SimpleNamespace(path=types.SimpleNamespace(exists=lambda value: False)),
            detection_model=types.SimpleNamespace(get_model=factory))
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'app.py', 'exec'), namespace)
        with self.assertRaisesRegex(FileNotFoundError, 'checkpoint'):
            namespace['get_inference_model']()
        factory.assert_not_called()

if __name__ == '__main__':
    unittest.main()
