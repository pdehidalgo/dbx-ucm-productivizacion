import base64
import logging
import os
from io import BytesIO
from mimetypes import guess_type
from typing import Union

import mlflow
from mlflow import metrics
from openai import AzureOpenAI, OpenAIError

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "image-description-prompts"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
DEFAULT_MODEL_DEPLOYMENT = "o4-mini"

_AZURE_OPENAI_CLIENT: AzureOpenAI | None = None


def get_azure_openai_client() -> AzureOpenAI:
    """
    Initializes and returns an Azure OpenAI client.

    Raises:
        ValueError: If required environment variables are not set.
        OpenAIError: If there's an issue creating the Azure OpenAI client.

    Returns:
        AzureOpenAI: An initialized Azure OpenAI client.
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")

    if not endpoint or not api_key:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables must be set."
        )

    global _AZURE_OPENAI_CLIENT
    if _AZURE_OPENAI_CLIENT is not None:
        return _AZURE_OPENAI_CLIENT

    try:
        _AZURE_OPENAI_CLIENT = AzureOpenAI(
            api_key=api_key,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=endpoint,
        )
        return _AZURE_OPENAI_CLIENT
    except OpenAIError as e:
        logger.error(f"Error creating Azure OpenAI client: {e}")
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while creating Azure OpenAI client: {e}"
        )
        raise


def _get_image_mime_type(image_input: Union[str, BytesIO]) -> str:
    if isinstance(image_input, BytesIO):
        mime_type = getattr(image_input, "mime_type", None)
        if isinstance(mime_type, str) and mime_type:
            return mime_type
    if isinstance(image_input, str):
        mime_type, _ = guess_type(image_input)
        if mime_type:
            return mime_type
    return "image/png"


def encode_image_to_base64(image_input: Union[str, BytesIO]) -> str:
    """
    Encodes an image from a file path or BytesIO object to a base64 string.

    Args:
        image_input: Path to the image file (str) or a BytesIO object.

    Returns:
        str: Base64 encoded string of the image.

    Raises:
        FileNotFoundError: If the image file does not exist.
        TypeError: If the image_input is of an unsupported type.
    """
    image_data = None
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        with open(image_input, "rb") as image_file:
            image_data = image_file.read()
    elif isinstance(image_input, BytesIO):
        image_input.seek(0)
        image_data = image_input.read()
    else:
        raise TypeError(
            "Unsupported image input type. Must be a file path (str) or BytesIO object."
        )

    return base64.b64encode(image_data).decode()


def evaluate_prompt(
    prompt: str,
    image: Union[str, BytesIO],
    model_deployment: str = DEFAULT_MODEL_DEPLOYMENT,
) -> Union[str, None]:
    """
    Evaluates a prompt against an image using the Azure OpenAI vision model.

    Args:
        prompt: The text prompt to send to the model.
        image: The image data, either as a file path (str) or BytesIO object.
        model_deployment: The name of the Azure OpenAI model deployment to use.

    Returns:
        str: The content of the model's response if successful, None otherwise.
    """
    try:
        encoded_image = encode_image_to_base64(image)
        mime_type = _get_image_mime_type(image)
        client = get_azure_openai_client()

        result = client.chat.completions.create(
            model=model_deployment,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
        )
        return result.choices[0].message.content
    except (OpenAIError, FileNotFoundError, TypeError, ValueError) as e:
        logger.error(f"Error during prompt evaluation: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during prompt evaluation: {e}")
        return None


def predict_image_description(
    prompts: list[str], img: Union[str, BytesIO] = "./images/error01.png"
) -> list[Union[str, None]]:
    """
    Generates descriptions for an image using a list of prompts.

    Args:
        prompts: A list of prompts to use for image description.
        img: The path to the image file.

    Returns:
        list[Union[str, None]]: A list of descriptions, one for each prompt.
        None is returned for prompts that failed.
    """
    return [evaluate_prompt(current_prompt, img) for current_prompt in prompts]


def run_experiments() -> None:
    """
    Runs MLflow experiments to evaluate different prompts for image description.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    prompts_to_evaluate = [
        "Describe in detail what you see in the image, including key visual elements, visible text, and any clues about the context or situation depicted.",
        "You are a technician specialized in operating systems. Analyze the image from a technical perspective and explain the type of error shown, possible causes, and suggestions to resolve it.",
        "Visually analyze the image and provide an objective interpretation of its content, considering both technical and communicative aspects. Describe who might be viewing this image and in what context.",
    ]

    expected_outputs = [
        """The image shows a Windows error screen commonly known as the "blue screen of death" (BSOD). The background is completely blue with white text.""",
        """As a technician, the image shows a blue error screen (BSOD) on a Windows operating system. The message indicates that the system encountered a serious problem and needs to restart. The stop code is: PAGE_FAULT_IN_NONPAGED_AREA, and the failing driver is: AcmeVideo.sys, which suggests the error is related to a custom video driver or third-party software tied to the graphics subsystem.""",
        """Visually, the blue screen indicates that the system has stopped due to a serious technical problem. The error is related to a custom video driver (AcmeVideo.sys), suggesting the user could be someone who uses graphics software or non-standard hardware.""",
    ]

    image_path = "./images/error01.png"
    if not os.path.exists(image_path):
        logger.warning(
            f"Image file for experiments not found at {image_path}. Skipping experiments."
        )
        return

    for prompt, target in zip(prompts_to_evaluate, expected_outputs):
        with mlflow.start_run():
            mlflow.log_param("prompt", prompt)

            def model_adapter(data):
                if isinstance(data, str):
                    prompts = [data]
                else:
                    prompts = list(data)
                return predict_image_description(prompts, image_path)

            try:
                mlflow.evaluate(
                    model=model_adapter,
                    data=[prompt],
                    targets=[target],
                    extra_metrics=[
                        metrics.latency(),
                        metrics.toxicity(),
                        metrics.rouge2(),
                        metrics.ari_grade_level(),
                    ],
                )
            except Exception as e:
                logger.error(
                    f"Error during MLflow evaluation for prompt '{prompt[:50]}...': {e}"
                )

    logger.info("Prompt evaluation completed. Check MLflow UI for results.")


def get_best_prompt(experiment_names: list[str]) -> Union[str, None]:
    """
    Retrieves the best performing prompt from MLflow runs based on predefined metrics.

    Args:
        experiment_names: A list of MLflow experiment names to search within.

    Returns:
        str: The best performing prompt, or None if no suitable runs are found.
    """
    order_by_cols = [
        "metrics.`toxicity/v1/mean` ASC",
        "metrics.`rouge2/v1/mean` DESC",
        "metrics.latency ASC",
    ]

    filter_query = 'status = "FINISHED"'

    try:
        runs = mlflow.search_runs(
            experiment_names=experiment_names,
            filter_string=filter_query,
            order_by=order_by_cols,
            max_results=1,
            output_format="list",
        )
        if runs:
            return runs[0].data.params.get("prompt")
        logger.warning(
            f"No finished MLflow runs found for experiments: {', '.join(experiment_names)}"
        )
        return None
    except Exception as e:
        logger.error(f"Error retrieving best prompt from MLflow: {e}")
        return None
