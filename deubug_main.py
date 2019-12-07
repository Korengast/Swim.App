from .pipeline.pipeline import Pipeline
from .pipeline.pipeline_executor import Pipeline_Executor


CLEAR_HISTORY = True
STEPS = [
    Pipeline.load_data,
    Pipeline.build_model,
    Pipeline.train_model,
    Pipeline.predict,
    Pipeline.publish_outputs
]


def main():
    pipeline_executor = Pipeline_Executor(CLEAR_HISTORY)
    pipeline_executor.execute(STEPS)

if __name__ == '__main__':
    main()