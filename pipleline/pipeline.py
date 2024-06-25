import kfp
from kfp import dsl

@dsl.component(base_image='python:3.8')
def data_generation_op() -> None:
    import subprocess
    subprocess.run(['python', 'data_generation.py'])

@dsl.component(base_image='python:3.8')
def model_training_op() -> None:
    import subprocess
    subprocess.run(['python', 'model_training.py'])

@dsl.component(base_image='python:3.8')
def tapestry_dht_op() -> None:
    import subprocess
    subprocess.run(['python', 'tapestry_dht.py'])

@dsl.component(base_image='python:3.8')
def smartthings_integration_op() -> None:
    import subprocess
    subprocess.run(['python', 'smartthings_integration.py'])

@dsl.pipeline(
    name='ML App Pipeline',
    description='Pipeline for ML application'
)
def ml_pipeline():
    data_gen_task = data_generation_op()
    model_train_task = model_training_op().after(data_gen_task)
    tapestry_dht_task = tapestry_dht_op().after(model_train_task)
    smartthings_integration_task = smartthings_integration_op().after(tapestry_dht_task)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(ml_pipeline, 'ml_pipeline.yaml')
