
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.step_collections import RegisterModel

def get_pipeline(region, role):
    processor = ScriptProcessor(image_uri="763104351884.dkr.ecr." + region + ".amazonaws.com/sklearn-processing:0.23-1",
                                command=["python3"],
                                role=role,
                                instance_type="ml.m5.xlarge",
                                instance_count=1)
    
    step_process = ProcessingStep(
        name="PreprocessData",
        processor=processor,
        code="pipeline/preprocess.py",
        outputs=[],
    )

    estimator = Estimator(
        image_uri="382416733822.dkr.ecr." + region + ".amazonaws.com/linear-learner:1",
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=f"s3://your-s3-bucket/output"
    )

    step_train = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={"train": "s3://your-s3-bucket/train.csv"}
    )

    step_register = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="SampleModelGroup",
    )

    pipeline = Pipeline(
        name="SampleMLPipeline",
        steps=[step_process, step_train, step_register]
    )

    return pipeline
