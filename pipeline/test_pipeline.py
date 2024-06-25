import responses
import kfp
from kfp.dsl import component
from kubernetes.client.models import V1EnvVar

@responses.activate
def test_my_pipeline():
    # Mocking Kubeflow Pipelines health check endpoint (optional for local testing)
    responses.add(
        responses.GET,
        "http://your-kubeflow-pipelines-ui-address/apis/v2beta1/healthz",
        json={"status": "ok"},
        status=200,
    )

    # Define components using the @component decorator
    @component(
        packages_to_install=["tapestry-dht", "smartthings-sdk"],  # Install required packages
        base_image="python:3.9",                               # Specify the base image
    )
    def data_generation():
        from data_generation import generate_data

        generate_data()

    @component
    def model_training(model_path: str):
        from model_training import train_model

        train_model(model_path)

    @component
    def tapestry_dht(tapestry_config: str):
        from tapestry_dht import setup_dht

        setup_dht(tapestry_config)

    @component
    def smartthings_integration(smartthings_token: str):
        from smartthings_integration import integrate_with_smartthings

        integrate_with_smartthings(smartthings_token)

    # Define the pipeline
    @kfp.dsl.pipeline(
        name="ML App Pipeline",
        description="Pipeline for ML application",
    )
    def ml_pipeline(
        model_path: str = "model.pkl",  # Example parameter with default
        tapestry_config: str = "tapestry_config.json",  # Example parameter with default
        smartthings_token: str = "",                    # Example parameter 
    ):

        # Use the component functions to define tasks
        data_gen_task = data_generation()
        model_training_task = model_training(model_path).after(data_gen_task)
        tapestry_dht_task = tapestry_dht(tapestry_config).after(model_training_task)
        smartthings_integration_task = smartthings_integration(smartthings_token).after(tapestry_dht_task)

    # Compile the pipeline
    kfp.compiler.Compiler().compile(ml_pipeline, "ml_pipeline.yaml")
    print("Pipeline compiled successfully.")


if __name__ == "__main__":
    test_my_pipeline()
