# Изменения в коде для поддержки компиляции TIDL версии 11_00_06_00


## TIDl не поддерживает динамические оси

Т.к. TIDL не поддерживает динамические оси, то при конвертации в onnx заккоментируем данную срточку. Также, оставим модель без постпроцессинга для анализа.

```python
torch.onnx.export(
    model,
    (data), #was (data, size)
    args.file_name,
    input_names=['images'], #was ['images', 'orig_target_size']
    output_names=['boxes', 'scores'], #was ['labels', 'boxes', 'scores']
    #dynamic_axes=dynamic_axes,
    opset_version=16,
    verbose=False)
```

Измененный код для построения модели

```python
class Model(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        #self.postprocessor = cfg.postprocessor.deploy()
        #print(self.postprocessor.deploy_mode)

    def forward(self, images):
        outputs = self.model(images)
        return outputs  #self.postprocessor(outputs, orig_target_sizes)

model = Model()
```